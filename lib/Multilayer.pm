package Multilayer;
# 多層パーセプトロン用モジュール といっても低層レベルの
# Perceptronモジュールを集約して、
# 誤差逆伝搬法（バックプロパゲーション）を試す
# 学習には全体を見渡さないと出来ないので。。。
# 最終的にはネットワーク経由でノードを分散させることも考える
# ノード側PerceptronモジュールのcalcReLU,calcStepメソッドを呼び出して、計算し、
# 活性化関数を分離した書き方に変更したので、->calcSum->ReLU()となる
# 重み付けは計算して、waitsメソッドでノードに登録していく
#
# Perceptronの収束確率が低いので、どこまでうまく動作するのか不明

use Carp;
use List::Util;
use Clone qw/ clone /;
use Data::Dumper;
use Scalar::Util qw/ weaken /;
#use DateTime;
use Time::HiRes qw / time /;
use feature 'say';

use FindBin;
use lib "$FindBin::Bin";
use Perceptron;
use Datalog;

use utf8;
binmode 'STDOUT' , ':utf8';


sub new {
    my $proto = shift;
    my $class = ref $proto || $proto;

    my $self = {};

       $self->{layer} = [];  # 層を表現する２次元で全体像を表す
       # {layer}->[0]が入力層、->[n]が出力層のイメージ
       #
       $self->{layer_count} = undef; # 層の数 配列の添字想定
       $self->{layer_member} = undef; # 層内のノード数
       $self->{input_count} = undef; # ノードへの入力数
       $self->{learn_rate} = undef; # 学習率を全体反映させる
       $self->{initdata} = undef; # layer_init時に指定された引数
       $self->{error_count} = {}; # 教育データのクラス毎にエラー回数
       $self->{learn_input} = undef; #バッチ単位の学習用データ1次元

       #  $self->{picup_cnt} = undef; # サンプルデータ数   $self->{initdata}->{batch}で登録される
       #  $self->{batch} = undef; # バッチのデータ数
       $self->{intre} = undef; # イテレーター数
       #  $self->{epoc} = undef; #エポック数
       $self->{all_learndata} = undef; # 学習用データの全てバッチ構成前
       $self->{interater} = undef ; # 学習用データをバッチ単位で配列にしたもの ARRAYref
                                # 標準化前の状態まで

       $self->{stat} = ""; # モジュールのステータス
                           # layer_inited : 初期化済  ->learn()　が動作する
                           # learned :　学習済   ->calc_multi()が動作する
			   #
       $self->{input} = undef ;
       $self->{learn_limit} = 2000;   # 学習データ1個に対して、waitsの更新を制限する。しかし、waitsに変化がないと抜けるのでlimitになっていない
       $self->{learn_finish} = {};  #学習が終わるためのチェックリスト　ハッシュでclassラベルをチェックする
       $self->{debug_flg} = 0; # 0:off 1:on 

       $self->{calc_multi_out} = undef; # clac_multiの結果を保持する　->lossで利用する
       $self->{old_layerwaits} = undef;
       $self->{new_layerwaits} = undef;
       $self->{fillARRAY} = [];
       $self->{backprobacation} = undef;
       $self->{adam_waits} = undef; # waits用 v,sを記録しておく adamで利用
       $self->{adam_bias} = undef; #bias用 backprobacation adamで利用
       $self->{act_func} = undef; # 活性化関数の微分をそれぞれ用意する
       $self->{adam_params} = { 
	                        mini_num => 1e-12,
	                        moment_beta => 0.9,
	                        rms_beta => 0.99,
			      };

       $self->{datalog_name} = undef; # Datalog db file name
       $self->{datalog} = undef;
       $self->{datalog_transaction} = 'off'; # トランザクションモード  off: autocommit on:transaction
       $self->{datalog_count} = undef; # カウンター用

       bless $self , $class;

    return $self;
}


sub layer_init {
    my $self = shift;
    # ノードを用意するために、レイヤー数とノード数を入力してもらう
    # ハッシュを想定、
    # waitを乱数で初期化する
    #

=pod
    #入力データ構造
     {  
        layer_member => [ 'ノード数 -1', ・・・ ],   階層毎のノード数 
        input_count => '入力数',  最初に入力するデータ量
        learn_rate => 0.34
        layer_act_func => [ 'ReLU' , 'ReLU' .... , 'Step' ],  # layer_memberと同じ項目数 layer毎の活性化関数を指定
        optimaizer => 'adam'.
        picup_cnt => 10000,  # 学習データのサンプル数
        batch => 50 ,   # バッチ学習のバッチあたりのデータ数
    　　epoc => 200 ,  # エポック数
     }
=cut
    if (!@_) {
        croak "Error layer_init input";
    }

    if ( $_[0] =~ /HASH/ ) {
        $self->{initdata} = $_[0];    
	$self->{layer_member} = $self->{initdata}->{layer_member};
	weaken($self->{layer_member});
	my @tmp = @{$self->{layer_member}};
	$self->{layer_count} = $#tmp;
	$self->{input_count} = $self->{initdata}->{input_count};
	undef @tmp;
	$self->{layer_act_func} = $self->{initdata}->{layer_act_func};
	weaken($self->{layer_act_func});
	if ( exists $self->{initdata}->{learn_rate} ) {
            $self->{learn_rate} = $self->{initdata}->{learn_rate};
	}
	

    } else {
        croak "initdata error!";
    }

    $self->{layer} = []; # 初期化　

    for (my $l=0;$l<=$self->{layer_count};$l++) {
        $self->{tmp}->{nodes} = [];
        for (my $n=0; $n<=$self->{layer_member}->[$l]; $n++) {
            push( @{$self->{tmp}->{nodes}} , Perceptron->new );
	    # waitsの初期化

	    # ローカルサブルーチンの定義 活性化関数毎に初期化方法を設定
	    my $subs->{ReLU} =  sub {  
		    my ($self , $l , $n ) = @_;

		    my $node_count = $self->{layer_member}->[$l] + 1;
		    if ( $l == 0 ) {
			# He初期化のためレイヤーのノード数を送る
			$self->{tmp}->{nodes}->[$n]->waitsinit($self->{input_count} , $node_count , 'He' );
			#$self->{tmp}->{nodes}->[$n]->waitsinit($self->{input_count});  # 乱数初期化
		    } else {
			$self->{tmp}->{nodes}->[$n]->waitsinit($self->{layer_member}->[$l-1] , $node_count , 'He' );  # 一つ前の階層のノード数
			#$self->{tmp}->{nodes}->[$n]->waitsinit($self->{layer_member}->[$l-1]);  #乱数初期化 
		    }
		    if ( defined $self->{learn_rate} ) {
			# 学習率が指定されていれば変更する
			$self->{tmp}->{nodes}->[$n]->learn_rate($self->{learn_rate});
			$self->{learn_limit} = (1 / $self->{learn_rate}); # limitは学習率の逆数
		    }
	    }; # sub ReLU

	    # ReLUのデバッグのためにNoneオプションを作成 -> 活性化関数無しに変更
	    $subs->{None} =  sub {  
		    my ($self , $l , $n ) = @_;

		    my $node_count = $self->{layer_member}->[$l] + 1; # 添字なので+1
		       $node_count += $self->{layer_member}->[$l + 1 ] + 1 if defined $self->{layer_member}->[$l + 1];  # 後ろのノード数も加える 
		    if ( $l == 0 ) {
			#my $node_count = $self->{layer_member}->[$l] + 1;
			$self->{tmp}->{nodes}->[$n]->waitsinit($self->{input_count} , $node_count , 'Xavier');
			#$self->{tmp}->{nodes}->[$n]->waitsinit($self->{input_count});  # 乱数初期化
		    } else {
			    #my $node_count = $self->{layer_member}->[$l] + 1;
			$self->{tmp}->{nodes}->[$n]->waitsinit($self->{layer_member}->[$l-1] , $node_count , 'Xavier');  # 一つ前の階層のノード数
			    #$self->{tmp}->{nodes}->[$n]->waitsinit($self->{layer_member}->[$l-1]);  #乱数初期化 
		    }
		    if ( defined $self->{learn_rate} ) {
			# 学習率が指定されていれば変更する
			$self->{tmp}->{nodes}->[$n]->learn_rate($self->{learn_rate});
			$self->{learn_limit} = (1 / $self->{learn_rate}); # limitは学習率の逆数
		    }
	    }; # sub None

	    $subs->{Step} = sub {
		    my ($self , $l , $n ) = @_;

		    if ( $l == 0 ) {
			$self->{tmp}->{nodes}->[$n]->waitsinit($self->{input_count});  # 乱数初期化
		    } else {
			$self->{tmp}->{nodes}->[$n]->waitsinit($self->{layer_member}->[$l-1]);  #乱数初期化 
		    }
		    if ( defined $self->{learn_rate} ) {
			# 学習率が指定されていれば変更する
			$self->{tmp}->{nodes}->[$n]->learn_rate($self->{learn_rate});
			$self->{learn_limit} = (1 / $self->{learn_rate}); # limitは学習率の逆数
		    }
	    }; # sub Step

	    $subs->{Sigmoid} = sub {
		    my ($self , $l , $n ) = @_;
		    # Xavier初期化

		    my $node_count = $self->{layer_member}->[$l] + 1; # 添字なので+1
		       $node_count += $self->{layer_member}->[$l + 1 ] + 1 if defined $self->{layer_member}->[$l + 1];  # 後ろのノード数も加える 
		    if ( $l == 0 ) {
			$self->{tmp}->{nodes}->[$n]->waitsinit($self->{input_count} , $node_count , 'Xavier' );  # 乱数初期化
		    } else {
			$self->{tmp}->{nodes}->[$n]->waitsinit($self->{layer_member}->[$l-1] , $node_count , 'Xavier' );  #乱数初期化 
		    }
		    if ( defined $self->{learn_rate} ) {
			# 学習率が指定されていれば変更する
			$self->{tmp}->{nodes}->[$n]->learn_rate($self->{learn_rate});
			#	$self->{learn_limit} = 100;
			$self->{learn_limit} = (1 / $self->{learn_rate}) ; # limitは学習率の逆数
		    }
	    }; # sub Sigmoid

	    $subs->{$self->{layer_act_func}->[$l]}->( $self , $l , $n ); 

        }
    push (@{$self->{layer}} , $self->{tmp}->{nodes});
    undef $self->{tmp};
    }
    
    # チェック比較時に利用する配列
    for my $l ( 0 .. $self->{layer_count} ){
        for my $n ( 0 .. $self->{layer_member}->[$l] ) {
            push(@{$self->{fillARRAY}->[$l]} , 1); 
        }
    }

    $self->{act_funcs}->{ReLU} = sub {
        my ($self , $l , $n ) = @_;
        my $second = undef;
	# 活性化関数がReLUのケース
	my $bias = $self->{layer}->[$l]->[$n]->bias(); 
	if ( $out->[$l]->[$n] >= $bias ) {
	    $second = 1; # 活性化関数 の微分 ReLU関数
	} else {
	    $second = 0;
	}
	return $second;
    }; # sub

# ReLUのデバッグ用None  -> 活性化関数無し
    $self->{act_funcs}->{None} = sub {
        my ($self , $l , $n ) = @_;
        my $second = 1;  # 常時１
=pod
			# 活性化関数がReLUのケース
			my $bias = $self->{layer}->[$l]->[$n]->bias(); 
			if ( $out->[$l]->[$n] >= $bias ) {    # 古い記述 outがすべて0になる Perceptron.pmを修正しているのでこの記述でも行けるのでは？
			##if ( $out->[$l]->[$n] > 0 ) {
			    $second = 1; # 活性化関数 の微分 ReLU関数
			} else {
			    $second = 0;
			}
=cut

	return $second;
    }; # sub

    $self->{act_funcs}->{Step} = sub {
        my ($self , $l , $n ) = @_;
        my $second = $out->[$l]->[$n];  # Step関数の微分
        return $second;
    }; # sub

    $self->{act_funcs}->{Sigmoid} = sub {
        my ($self , $l , $n ) = @_;
        my $bias = $self->{layer}->[$l]->[$n]->bias();
        my $calc_sum = $self->{layer}->[$l]->[$n]->calc_sum();
           $calc_sum += $bias;
        my $second = ( 1 - ( 1 / (1 + exp( -$calc_sum)))) * ( 1 / (1 + exp( -$calc_sum)));  # sigmoid関数の微分
        return $second;
    }; # sub

    $self->{stat} = "layer_inited";
    undef $subs;
}

sub takeover {
    my $self = shift;
    # dump_structureからwaitsとbiasを引き継ぐ
    # dump hashをそのまま入力されることを想定
    # 前提はlayer_initが済んでいること
    # dump_structure.txtをrequireで読み込んで引数に利用する

    if (@_) {
        if ($_[0] =~ /HASH/) {
            if ( exists $_[0]->{bias} and $_[0]->{waits} ) {
                $self->{takeover} = $_[0];
	    } else {
                croak "Missing data in dump";
	    }
	} else {
            croak "miss dump data!";
	}
    }

    if ($self->{stat} ne 'layer_inited') {
        croak "not layer init";
    }


    my @layer = @{$self->{layer}};
    for (my $l=0; $l<=$#layer; $l++) {
	my @nodes = @{$layer[$l]};
        for (my $n=0; $n<=$#nodes; $n++) {
	    $nodes[$n]->waits($self->{takeover}->{waits}->{$l}->{$n});
	    $nodes[$n]->bias($self->{takeover}->{bias}->{$l}->{$n});
	}
    }
    undef @layer;
    undef @nodes;
    undef $self->{takeover};
}

sub disp_waits {
    my $self = shift;
    # 引数が無ければ、全体を表示する

    # 引数に数値があればレイアー番号として、限定表示
    if (@_) {
        if ($_[0] =~ /^\d?/ ) {
            my @layer = @{$self->{layer}};
	    my @nodes = @{$layer[$_[0]]};
            for (my $n=0; $n<=$#nodes; $n++) {
	        my $waits = $nodes[$n]->waits();
	        my $bias = $nodes[$n]->bias();
		my $learn_rate = $nodes[$n]->learn_rate();

                say "Layer: $_[0] Node: $n";
	    
	        print "waits: $_ " for @{$waits};
	        say "";
	        say "bias: $bias";
		say "learn_rate: $learn_rate";
	        say "-----------------------";

	        undef $waits;
	        undef $bias;
	        undef $learn_rate;
	    }
	    undef @nodes;
	    undef @layer;
        }
    } else {

    my @layer = @{$self->{layer}};
    for (my $l=0; $l<=$#layer; $l++) {
	my @nodes = @{$layer[$l]};
        for (my $n=0; $n<=$#nodes; $n++) {
	    my $waits = $nodes[$n]->waits();
	    my $bias = $nodes[$n]->bias();
            my $learn_rate = $nodes[$n]->learn_rate();

            say "Layer: $l Node: $n";
	    
	    print "waits: $_ " for @{$waits};
	    say "";
	    say "bias: $bias";
            say "learn_rate: $learn_rate";
	    say "-----------------------";

	    undef $waits;
	    undef $bias;
	    undef $learn_rate;
	}
        undef @nodes;
    }
    undef @layer;

    } # else
}

sub dump_structure {
    my $self = shift;
    # file dump data
    # make "dump_structure.txt" HASH ref
    # 引数にcheckが在ると、ハッシュデータを戻す
    $self->{dump_check} = 0;
    if (@_) {
        if ($_[0] eq 'check' ) {
            $self->{dump_check} = 1;
	}
    }

    my $waitsdump = {};  # $waitsdump->{layer}->{node} refARRAY
    my $biasdump = {};  # $biasdump->{layer}->{node} scalar
    my $learn_rate_dump = {}; 

    my @layer = @{$self->{layer}};
    for (my $l=0; $l<=$#layer; $l++) {
	my @nodes = @{$layer[$l]};
        for (my $n=0; $n<=$#nodes; $n++) {
	    my $waits = $nodes[$n]->waits();
	    my $bias = $nodes[$n]->bias();
            my $learn_rate = $nodes[$n]->learn_rate();

	    $waitsdump->{$l}->{$n} = $waits;
            $biasdump->{$l}->{$n} = $bias;
	    $learn_rate_dump->{$l}->{$n} = $learn_rate;

	    undef $waits;
	    undef $bias;
	    undef $learn_rate;
	}
	undef @nodes;
    }
    undef @layer;
 
    #my $dt = DateTime->now();
    my $dt = time();

    my $dumpdata = { 
                    DateTime => $dt,
                    layer_init => $self->{initdata} ,
		    waits => $waitsdump,
		    bias => $biasdump,
                    learn_rate => $learn_rate_dump,
                   };

    if ( $self->{dump_check} == 1 ) {

        return $dumpdata;

    } else {

    open (my $fh , '> ./dump_structure.txt');

        say $fh "#dump data structure";
        say $fh "# hash data key 'layer_init' 'waits' 'bias' 'learn_rate'";
        say $fh "# layerdata->{layer}->{node} ";
        say $fh "# waitsdump: waits data to node refARRAY";
        say $fh "# layer_init: layer_init set data";
        say $fh "# bias: need set for perceptron";
        say $fh "# learn_rate: extra data";

        print $fh Dumper($dumpdata);

        close($fh);
    }

    undef $dumpdata;
    undef $dt;
    undef $waitsdump;
    undef $biasdump;
    undef $learn_rate_dump;
} # dump_structure

sub all_learndata {
    my $self = shift;
    # 学習用データをARRAYrefでセットするセッター
    # データ構造
    # $sample = { input => [] , class => [] }; これがARRAYで入っている1次元構造

    if (@_) {
        if ( $_[0] =~ /ARRAY/ ) {
            $self->{all_learndata} = clone($_[0]);	
        } else {
        croak "all_learndata invarid data type";
	}	
    } else {
        croak "no input error!";
    }
}

sub prep_learndata {
    my $self = shift;
    # エポック単位のデータを抽出する
    # picup_cnt数のサンプルを抽出して、
    # バッチ数分のARREYrefをイテレーターに収納する

    if (! defined $self->{all_learndata} ) {
        croak "no all_learndata!";
    }

    srand();

    undef $self->{intereter};  # 初期化
    $self->{interater} = [];

    my $batch = $self->{initdata}->{batch};
    my $picup_cnt = $self->{initdata}->{picup_cnt};
    my $intre = int( $picup_cnt / $batch );
       $self->{intre} = $intre;
    my $epoc = $self->{initdata}->{epoc};

    my $learndata = [];

    my @tmp = @{$self->{all_learndata}};
    my $data_cnt = $#tmp;
    undef @tmp;

    for my $cnt ( 1 .. $picup_cnt ) {
        my $choice = int(rand($data_cnt));
        my $sample = clone($self->{all_learndata}->[$choice]);
	# データ構成のチェック
        if (( exists $sample->{input} ) && ( exists $sample->{class})){
            push(@{$learndata} , $sample);
	} else {
            croak "invarid data structure!";
	}

    } #for cnt

    #バッチに分割
    for my $i ( 1 .. $intre ) {
        my $tmp = [];
        for my $j ( 1 .. $batch ) {
            push(@{$tmp} , shift(@{$learndata}));
        }
        push(@{$self->{interater}} , $tmp);
        undef $tmp;
    }

    undef $learndata;
}

sub get_interater {
    my $self = shift;
    # $self->{interater}のゲッター　標準化前のデータ
    # prep_learndataを実行しないとundefか空配列が戻る
    return $self->{interater};
}

sub input_layer {
    my $self = shift;
    #入力層を標準化する　0-1にまとめる
    # {input}と{class}を変更する
    # prep_learndataが済んでいること

    if (! defined $self->{intre} ) {
        croak "no action prep_learndata!!!";
    }

        my @list_input = (); #全てのinput
        my @list_class = (); #全てのclass
        for my $batch (@{$self->{interater}}) {
            for my $sample (@{$batch}) {
                push(@list_input , @{$sample->{input}});
                push(@list_class , @{$sample->{class}});
            }
        }
        my $min_input = List::Util::min(@list_input);
        my $max_input = List::Util::max(@list_input);
        my $min_class = List::Util::min(@list_class);
        my $max_class = List::Util::max(@list_class);

        undef @list_input;
        undef @list_class;

=pod
        my $input_offset = undef;
        my $input_width = undef;

        if ($min_input < 0 ) {
           $input_offset = abs($min_input);
           $input_width = $max_input + $input_offset;
        } else {
           $input_offset = $min_input;
           $input_width = $max_input - $min_input;
        }

        my $class_offset = undef;
        my $class_width = undef;

        if ($min_class < 0 ) {
            $class_offset = abs($min_class);
            $class_width = $max_class + $class_offset;
        } else {
            $class_offset = $min_class;
            $class_width = $max_class - $min_class;
        }
=cut

        #標準化   コメントしているのは正規化なのでとりあえず標準化で
	my $interater = [];
        for (my $i=0; $i <= $self->{intre} - 1; $i++) {
            for (my $j=0 ; $j <= $self->{initdata}->{batch} - 1; $j++) {
                    #  @{$interater->[$i]->[$j]->{input}} = map { ($_ + $input_offset ) / $input_width } @{$interater->[$i]->[$j]->{input}};
                    #  @{$interater->[$i]->[$j]->{class}} = map { ($_ + $class_offset ) / $class_width } @{$interater->[$i]->[$j]->{class}};
                @{$interater->[$i]->[$j]->{input}} = map { ($_ - $min_input ) / ($max_input - $min_input )} @{$self->{interater}->[$i]->[$j]->{input}};
                @{$interater->[$i]->[$j]->{class}} = map { ($_ - $min_class ) / ($max_class - $min_class )} @{$self->{interater}->[$i]->[$j]->{class}};
            }
        }
	# $self->{interater} は標準化前の状態
	# $interaterは標準化後の状態

        return $interater;
} # input_layer

sub learn {
    my $self = shift;
    # perceptronと同じフォーマットでサンプルデータを受け入れる。
    # ただし、論理は0 or 1に変更される。
    # サンプルクラス毎に最低1回収束するところで学習済みとフラグを立てる。
    # 振り分けに失敗する場合は、外部からサンプルを替えて学習を続けることで収束も可能になる
    #
    # 手順は勾配降下法
    #

    if (! $self->{stat} eq "layer_inited") {
        croak "Still init ...";
    }

    if (!@_ ) {
        croak "no larning data";
    }

    if (@_) {
        if ($_[0] =~ /ARRAY/) {
            $self->{learn_input} = $_[0];  # learn内のみ変数  バッチ単位の一次元データ
	} else {
            croak "input data format not match!";
	}
    } 

    my $debug = $self->{debug_flg}; # 0: off 1: on
    my $hand = 0; # 0: off 1: on  手動実行時にほしい表示 収束するか傾向を見る場合

    #  $self->datalog_init();  # epoc毎に取得するため、外部に移動 ここに書くとバッチ単位で実行されるので


=pod # 下記　datalog_snapshotに置き換え
    # 初期化データをダンプ形式で記録するハッシュなので構造が違う
    my $start_waits = $self->dump_structure('check');
    my $start_waits_strings = Dumper $start_waits;
    $self->{datalog}->addlog($start_waits_strings);
    undef $start_waits;
    undef $start_waits_strings;

    $self->{datalog}->begin_work() if $self->{datalog_transaction} eq 'on';
=cut

    #$self->datalog_snapshot();


=pod  # whileをコメントしたのでここもコメント ->layer_initに移動
    # チェック比較時に利用する配列
    my $fillARRAY = [];
    for my $l ( 0 .. $self->{layer_count} ){
        for my $n ( 0 .. $self->{layer_member}->[$l] ) {
            push(@{$fillARRAY->[$l]} , 1); 
        }
    }
=cut

    my $loop = 0;
    # 入力して各層を計算していく
    for my $sample (@{$self->{learn_input}}) {
        $loop++;

        my $sample_flg = 1;
	my $sample_count = 0;
#        while ( $sample_flg ) {  

=pod	# whileをコメントしたのでここもコメントに
            if ($sample_count >= $self->{learn_limit} ) {
                &::Logging("learn limit over!  $sample_count");
		my $dump_strings = Dumper $sample;
		&::Logging("dump : $dump_strings ");
		undef $dump_strings;
	       # $self->{error_count}->{@{$sample->{class}}}++; # クラス毎にエラーをカウント
		$self->{error_count}->{"@{$sample->{class}}"}++; # クラス毎にエラーをカウント
		$sample_flg = 0;
	        
		# exit;
	    }
            $sample_count++;
=cut

            &::Logging("Loop: $loop start ------------------") if $debug == 1;

	    #my $out = []; #3階層に成るように各ノードの出力結果  $out->[レイヤー]->[ノード]

	    $self->{tmp} = [];
	    @{$self->{tmp}} = @{$sample->{input}}; # デリファレンスでリークを回避

	    # None に前段があると出力が0になる件
	    #	    &::Logging("sample: ") if $debug == 1;
	    #print Dumper $sample if $debug == 1;
	    #&::Logging("tmp: ") if $debug == 1;
	    #print Dumper $self->{tmp} if $debug == 1;

            $self->input($self->{tmp});
	    my $out = $self->calc_multi('learn');
	       # $self->{calc_multi_out}にも同じ値が入っている

	    undef $self->{tmp};

=pod   # biasのupdateに利用する目的で作ったが間違い　使わない
            # calc_sumを集計しておく ノード毎の活性化関数前の値
	    # calc_sum()はアクセサー
	    # 2乗誤差計算で使わなかった
	    my $calc_sum = [];
            for my $l ( 0 .. $self->{layer_count} ) {
                for my $n ( 0 .. $self->{layer_member}->[$l] ) {
		    #my $tmp = $self->{layer}->[$l]->[$n]->calc_sum(); 
		    #$calc_sum->[$l]->[$n] =  $tmp ; #scar経由なら値はコピーされる、リファレンスされない
		    $calc_sum->[$l]->[$n] = $self->{layer}->[$l]->[$n]->calc_sum(); 
                }
            }
=cut

=pod   # 学習の度にチェックしていたがパーセプトロンと違って多層の場合はここで判定しない
	    # 出力層の結果をsampleのclassラベルと比較する
	    my $outstring = join ("" , @{$out->[$self->{layer_count}]});
	    my $sampleclassstring = join ("" , @{$sample->{class}});

	    &::Logging("DEBUG: outstring: $outstring") if $debug == 1;
	    &::Logging("DEBUG: sampleclassstring: $sampleclassstring") if $debug == 1;

	    $self->{learn_finish}->{$sampleclassstring} ||= 0; # 無ければ0をセット  何度も上書きされるので注意
	    &::Logging("DEBUG: learn_finish: $sampleclassstring $self->{learn_finish}->{$sampleclassstring}") if $debug == 1 ;

	    if ($sampleclassstring eq $outstring) {
               # 出力結果が一致したら

	       &::Logging("loop: $loop finish!") if $hand == 1;
               $sample_flg = 0;
	       $self->{learn_finish}->{$sampleclassstring} = 1;  # sampleclassstringが成功していること

	    } else {
                # 結果が一致しなかったら
    # 判定して修正する意味は無かったらしいのでlearn_limitをepocとして利用する
=cut
                #重み付けの更新 

=pod
		# ->lossに変更
	        #2乗誤差関数  (Sigma(出力層 - sample_class)^2 )/2 
		my $esum = 0;
                for my $node ( 0 .. $self->{layer_member}->[-1]) {
                    $esum += ($out->[$self->{layer_count}]->[$node] - $sample->{class}->[$node] ) ** 2;
		}
		$esum = $esum / 2;
=cut

		#my $esum = $self->loss($out);
		#&::Logging("DEBUG: 誤差関数 $esum") ; # if $hand == 1;

                # 現在のwaitsを取得する biasを除く
		$self->{old_layerwaits} = undef;
		$self->{old_layerwaits} = [];  # 3次元配列
		for my $l ( 0 .. $self->{layer_count}) {
                    for my $node ( 0 .. $self->{layer_member}->[$l]) {
                        my $waits_old = $self->{layer}->[$l]->[$node]->waits();
                        push(@{$self->{old_layerwaits}->[$l]} , $waits_old ); #waitsはノード単位
		    }	
                } 
		# 現在のbiasを取得する
		$self->{old_layer_bias} = undef;
		$self->{old_layerbias} = [];  # 3次元配列
		for my $l ( 0 .. $self->{layer_count}) {
                    for my $node ( 0 .. $self->{layer_member}->[$l]) {
                        my $bias_old = $self->{layer}->[$l]->[$node]->bias();
                        push(@{$self->{old_layerbias}->[$l]} , $bias_old ); #biasはノード単位
		    }	
                } 

		&::Logging("DEBUG: old_layerwaits") if $debug == 1;
#		print Dumper $self->{old_layerwaits} if $debug == 1;
		&::Logging("DEBUG: old_layerbias") if $debug == 1;
#		print Dumper $self->{old_layerbias} if $debug == 1;
		
		$self->{new_layerwaits} = undef;
		$self->{new_layerbias} = undef; 
		$self->{backprobacation} = undef;

		$self->{new_layerwaits} = [];
		$self->{new_layerbias} = [];  # logの為だけに取得 計算上は不要
		$self->{backprobacation} = [];
                # 出力層から順に処理する カウントダウンループ  l:レイヤー n:ノード w: wait入力層とその他でループが違う
		for (my $l=$self->{layer_count}; $l>=0; $l--) {
                    for (my $n=$self->{layer_member}->[$l]; $n>=0; $n--) {
                        my $learn_rate = $self->{layer}->[$l]->[$n]->learn_rate();
                        my $waits_delta = [];

			# waitsループは入力層とそれ以外でループが分かれる
			if ( $l == 0 ) {
			    for my $w ( 0 .. $self->{input_count} ) {
				my ( $first , $second , $third ) = ( undef , undef , undef );
		        # 入力層
			        for my $nsum ( 0 .. $self->{layer_member}->[$l+1] ) {
                                    $first += $self->{backprobacation}->[$l+1]->[$nsum]->[$n]->{first} * $self->{backprobacation}->[$l+1]->[$nsum]->[$n]->{second} * $self->{new_layerwaits}->[$l+1]->[$nsum]->[$n]; 
				}

			        $second = $self->{act_funcs}->{$self->{layer_act_func}->[$l]}->( $self , $l , $n );

				$third = $sample->{input}->[$w]; # 入力値 ########
				$self->{backprobacation}->[$l]->[$n]->[$w] = clone({ first => $first , second => $second , thire => $third }); 
				&::Logging("DEBUG: learn_rate: $learn_rate first: $first second: $second third $third ") if $debug == 1;
				#my $tmp = $self->{old_layerwaits}->[$l]->[$n]->[$w] - ( $learn_rate * $self->{backprobacation}->[$l+1]->[$self->{layer_member}->[$l+1]]->[$n]->{first} * $self->{backprobacation}->[$l+1]->[$self->{layer_member}->[$l+1]]->[$n]->{second} * $third );

                                my $tmp = $self->optimaizer($l , $n , $w);
                                push (@{$waits_delta} , $tmp); #調整したwaits

				undef $tmp;
                            } # for 入力層 w #################################
		        } else {
                            # 出力層
			    for my $w ( 0 .. $self->{layer_member}->[$l-1] ) {   # waitsは順方向ループ 前の層のノード数がwaitsの数
			        &::Logging("DEBUG: layer: $l node: $n waits: $w") if $debug == 1;

				my ( $first , $second , $third ) = ( undef , undef , undef );

                                if ($l == $self->{layer_count}) {
		                # 出力層の重み付け調整 
			            $first = $out->[$l]->[$n] - $sample->{class}->[$n];  # 誤差関数の偏微分->今回の出力からクラスラベル差

				    #  活性化関数に寄って変更される
				    #  $second = $out->[$l]->[$n];   # 活性化関数の偏微分 ->出力値そのまま
			            $second = $self->{act_funcs}->{$self->{layer_act_func}->[$l]}->( $self , $l , $n );

				    # $first * $second が一般のgradとされている値

                                    $third = $out->[$l-1]->[$w];   #入力から得られた結果の偏微分 -> 前の層からの入力
				    $self->{backprobacation}->[$l]->[$n]->[$w] = clone({ first => $first , second => $second , third => $third }); # 次の層の計算で利用される

				    &::Logging("DEBUG: learn_rate: $learn_rate first: $first second: $second third $third ") if $debug == 1;
				    #my $tmp = $self->{old_layerwaits}->[$l]->[$n]->[$w] - ( $learn_rate * $first * $second * $third ); 
                                    my $tmp = $self->optimaizer($l , $n , $w);

                                    push (@{$waits_delta} , $tmp);  # 調整したwaits 

				    undef $tmp;
			        } elsif ( $l < $self->{layer_count} ) {
                                # 中間層
			        # waitsは後層すべてのノードに影響するので、合計が必要になる waitsの添字は現在のノード番号
			            for my $nsum ( 0 .. $self->{layer_member}->[$l+1] ) {
				        $first += $self->{backprobacation}->[$l+1]->[$nsum]->[$n]->{first} * $self->{backprobacation}->[$l+1]->[$nsum]->[$n]->{second} * $self->{new_layerwaits}->[$l+1]->[$nsum]->[$n]; 
				    }
				    # $act_funcsに置き換わり 指定された活性化関数による
			            $second = $self->{act_funcs}->{$self->{layer_act_func}->[$l]}->( $self , $l , $n );
                                    $third = $out->[$l-1]->[$w];  # 前の層からの入力  ノードの番号はwaitsの添字

				    $self->{backprobacation}->[$l]->[$n]->[$w] = clone({ first => $first , second => $second , third => $third }); # 次の回の計算で利用される

				    &::Logging("DEBUG: learn_rate: $learn_rate first: $first second: $second third $third ") if $debug == 1;
=pod
				    # 一つ後層の計算結果を利用する
                                    my $theta = undef;
				    my $iota = undef;
                                    for my $node ( 0 .. $self->{layer_member}->[$l+1]) {
                                       $theta += $self->{backprobacation}->[$l+1]->[$node]->[$n]->{first};   # 今の$nが後ろのwaitの添字
				       $iota += $self->{backprobacation}->[$l+1]->[$node]->[$n]->{second};
                                    }
                                    my $tmp = $old_layerwaits->[$l]->[$n]->[$w] - ( $learn_rate * $theta * $iota * $third );
=cut
				    # 上を1行で置き換えた biasから派生して考え違いをしていた　waitsが含まれていない
				    #my $tmp = $self->{old_layerwaits}->[$l]->[$n]->[$w] - ( $learn_rate * $first * $second * $third ); 
				    my $tmp = $self->optimaizer($l , $n , $w);

                                    push (@{$waits_delta} , $tmp); #調整したwaits
                                    
                                    undef $tmp;
			       }
                            } # for $w  ############################################
		        } # if $l==0 esle

			# biasの更新 (ここはノード単位で1回)
			$self->optimaizer($l , $n , undef , $sample );  #waitは無いので入力しないとbiasの更新に成る

=pod   # optimaizerに移動
			# biasの更新 (ここはノード単位で1回)
			if ($l == $self->{layer_count} ) {
			    #　出力層
			    my $iota = $self->{act_funcs}->{$self->{layer_act_func}->[$l]}->( $self , $l , $n );
			    my $bias = $self->{layer}->[$l]->[$n]->bias();
			    #my $tmp = $bias - ( $learn_rate *  ($out->[$l]->[$n] - $sample->{class}->[$n]) * $iota * $out->[$l-1]->[$n]); # biasが前段の入力を得ているのがおかしいかもしれない
			    my $tmp = $bias - ( $learn_rate *  ($out->[$l]->[$n] - $sample->{class}->[$n]) * $iota * 1); 
			    $self->{layer}->[$l]->[$n]->bias($tmp);
			    $self->{new_layerbias}->[$l]->[$n] = $tmp;
			    &::Logging("DEBUG: OUTLAYER learn_rate: $learn_rate bias: $bias iota: $iota new_bias: $tmp ") if $debug == 1;
                            
                            undef $tmp;
			    undef $bias;
			    undef $iota;
		        } elsif ( $l < $self->{layer_count} ) {
			    # 中間層
			    my $bias = $self->{layer}->[$l]->[$n]->bias();
			    say "l: $l n: $n back_n: $self->{layer_member}->[$l+1] " if $debug == 1;
			    my $theta = undef;
                            my $iota = undef;
                            for my $node ( 0 .. $self->{layer_member}->[$l+1] ) {
				    # 一つ後ろの層のwaitsの添字が現在のノード番号になる
                                $theta += $self->{backprobacation}->[$l+1]->[$node]->[$n]->{first}; 
				$iota += $self->{backprobacation}->[$l+1]->[$node]->[$n]->{second};
                            }
			    my $tmp = $bias - ( $learn_rate * $theta * $iota * 1 ); # biasなので重み付けは1
                            $self->{layer}->[$l]->[$n]->bias($tmp);
			    $self->{new_layerbias}->[$l]->[$n] = $tmp;
			    &::Logging("DEBUG: MIDDLELAYER learn_rate: $learn_rate bias: $bias iota: $iota theta: $theta new_bias: $tmp ") if $debug == 1;

			    undef $tmp;
			    undef $bias;
			}
=cut

                        # ノードのwaitsを保持する
                        $self->{new_layerwaits}->[$l]->[$n] = clone($waits_delta); 

			&::Logging("DEBUG: backprobacation l: $l n: $n") if $debug == 1 ;
#                        print Dumper $self->{backprobacation} if $debug == 1;
		    } # for $n


		} # for $l

		&::Logging("DEBUG: new_layerwaits") if $debug == 1;
#		print Dumper $self->{new_layerwaits} if $debug == 1;

                # new_layerwaitsに値が入っていることを確認して、
		for my $l ( 0 .. $self->{layer_count} ) {
                    for my $n ( 0 .. $self->{layer_member}->[$l] ) {
			if ( $l == 0 ) {
			    # 入力層
                            for my $w ( 0 .. $self->{input_count}) {
                                if ( ! defined $self->{new_layerwaits}->[$l]->[$n]->[$w] ) {
                                    croak "new_layerwaits undef detected!! l: $l n: $n w: $w";
				    exit;
			        }
                            }
			} else {
			    # 中間層、出力層
                            for my $w ( 0 .. $self->{layer_member}->[$l-1] ) {
                                if ( ! defined $self->{new_layerwaits}->[$l]->[$n]->[$w] ) {
                                    croak "new_layerwaits undef detected!! l: $l n: $n w: $w";
				    exit;
			        }
                            }
		        }
                    }
		}
		
		# 各ノードにwaitsを再設定する
		for my $l ( 0 .. $self->{layer_count} ) {
                    for my $n ( 0 .. $self->{layer_member}->[$l] ) {
                        $self->{layer}->[$l]->[$n]->waits($self->{new_layerwaits}->[$l]->[$n]);
                    }
		}


=pod   # データ量が大きくなるのでコメントアウト中
		my $new_structure = { 
			              layer_init => $self->{initdata},
			              waits => $self->{new_layerwaits},
			              bias => $self->{new_layerbias},
				      node_out => $out,
				      node_sum => $calc_sum,
			            }; 

		my $new_structure_strings = Dumper $self->{new_structure};

		if ($self->{datalog_transaction} eq 'on' ) {

		    # トランザクションモード
                    $self->{datalog_count}++;
		    $self->{datalog}->addlog($new_structure_strings);
		    if ( $self->{datalog_count} >= ($self->{learn_limit} / 10) ) {
                        $self->{datalog}->commit();
			$self->{datalog}->begin_work(); # commitするとautoComitに戻るので、もう一回
                        $self->{datalog_count} = 0;
		    }
	        } elsif ($self->{datalog_transaction} eq 'off' ) {
		    # autoCommit
		    $self->{datalog}->addlog($new_structure_strings);
	        }

		undef $new_structure;
=cut

		&::Logging("DEBUG: loop $loop Change waits value  ------------------------") if $hand == 1;
		&::Logging("DEBUG: loop $loop Retry! $sample_count ------------------------") if $debug == 1;

#		$self->waitsChangeCheck($loop); # 思ったような動作にならないので封印

=pod # whileをコメントしたのでここもコメント  ->waitsChangeCheck()に移動
                # waitsの値が変化していないとループを抜ける仕組み
                my $check = [];
                for my $l ( 0 .. $self->{layer_count} ) {
                    for my $n ( 0 .. $self->{layer_member}->[$l] ) {
                        my $oldwaitsstring = join ("" , @{$old_layerwaits->[$l]->[$n]});
			my $newwaitsstring = join ("" , @{$new_layerwaits->[$l]->[$n]});
                        if ( $oldwaitsstring eq $newwaitsstring ) {
                            push( @{$check->[$l]} , 1);  # 一致
			} else {
                            push( @{$check->[$l]} , 0);  # 不一致
			}
                    }
                }
=cut

		undef $self->{new_layerwaits};

=pod # whileをコメントしたのでこの処理もコメントする  ->waitsChangeCheck()に移動
		
                for my $l ( 0 .. $self->{layer_count} ) {
                    my $checkstring = join ("" , @{$check->[$l]}); 
		    my $fillstring = join ("", @{$self->{fillARRAY}->[$l]});

                    if ($checkstring eq $fillstring) {
			    # レイヤー毎にチェックする
                        &::Logging("DEBUG: loop: $loop waits no change!!! layer $l sample_count: $sample_count") ; #if $hand == 1;
			$sample_flg = 0;
		    } 
                }
=cut
		
=pod   # 値が変化しなくなるのでコメントアウト
                # biasの値が変化していないとループを抜ける仕組み
                $check = []; # 再度初期化
                for my $l ( 0 .. $self->{layer_count} ) {
                    for my $n ( 0 .. $self->{layer_member}->[$l] ) {
                        my $oldbiasstring = join ("" , @{$self->{old_layerbias}->[$l]->[$n]});
			my $newbiasstring = join ("" , @{$self->{new_layerbias}->[$l]->[$n]});
                        if ( $oldwaitsstring eq $newwaitsstring ) {
                            push( @{$check->[$l]} , 1);  # 一致
			} else {
                            push( @{$check->[$l]} , 0);  # 不一致
			}
                    }
                }
=cut

		undef $self->{new_layerbias};

=pod  # 値が変化しなくなるのでコメント
                for my $l ( 0 .. $self->{layer_count} ) {
                    my $checkstring = join ("" , @{$check->[$l]}); 
		    my $fillstring = join ("", @{$fillARRAY->[$l]});

                    if ($checkstring eq $fillstring) {
			    # レイヤー毎にチェックする
                        &::Logging("DEBUG: loop: $loop bias no change!!! layer $l sample_count: $sample_count") if $hand == 1;
			$sample_flg = 0;
		    } 
                }
		undef $check;
=cut

#            } # if sampleclassstring      # if文をコメントしたのでこれもコメントする

#       } # while  # whileをコメント

        undef $out;
    } # for sample 

    #    $self->{datalog}->commit() if $self->{datalog_transaction} eq 'on';

=pod # 学習の度にチェックして完了を{stat}に記録するつもりで用意したが、これは使わない
    &::Logging("DEBUG: learn_finish data") if $debug == 1;
    # class毎の学習が完了したか目視用のDump
    &::Logging("class learn check dump self->{learn_finish}") if $hand == 1;
#    print Dumper $self->{learn_finish} if $hand == 1;
    # エラー率
    &::Logging("error_count  self->{error_count}") if $hand == 1;
#    print Dumper $self->{error_count} if $hand == 1;

    # class毎に成功しているかチェックする
    my $finish_cnt = 0;
    my $finish_loop = 0;
    for my $key ( keys %{$self->{learn_finish}} ) {
        $finish_loop++;
        if ($self->{learn_finish}->{$key} == 1 ) {
            $finish_cnt++;
        }
    }
    if ($finish_cnt == $finish_loop ) {

	    # ここに指定されたサンプルデータに基づいて、再度判定を行えば、
	    # 成功するまでループさせる処理を追加出来る。
	    # しかし、同じサンプルのまま、繰り返して、過学習になると、別の意味で使えない。

        $self->stat('learned');
    }
=cut

    # optimaizer処理もここでは終わっている想定
    undef $self->{learn_input};
    undef $self->{adam_waits};
    undef $self->{adam_bias};
    undef $self->{backprobacation};
    undef $self->{old_layerwaits};
    undef $self->{old_layerbias};
    undef $self->{new_layerwaits};
    undef $self->{new_layerbias};

    return $self->{stat};  # 完了ならlearndでなければlayer_initedが返る
}

sub loss {
    my ( $self , $intre ) = @_;
    # ->clac_multi()の結果を受け取って誤差関数の結果を返す
    # 直前のcalc_multiまたはlearnの直後を想定している
    # バッチ単位、epoc単位のどちらか

        if ((! defined $self->{calc_multi_out}) || (! defined $intre )) {
            &::Logging("INFO: calc_multi_out or sample undef...");
	    return;
	}

	my $sample = $intre->[-1]; #バッチの一番最後の値

	#my $out = $self->{calc_multi_out};
        #一旦計算してから判定する	
	$self->input($sample->{input});
	$self->calc_multi('learn');


        #2乗誤差関数  (Sigma(出力層 - sample_class)^2 )/2 
        my $esum = 0;
        for my $node ( 0 .. $self->{layer_member}->[-1]) {
            $esum += ($self->{calc_multi_out}->[$self->{layer_count}]->[$node] - $sample->{class}->[$node] ) ** 2;
	}
	$esum = $esum / 2;

	undef $out;
	undef $sample;
	undef $intre;

	return $esum;
}

sub waitsChangeCheck {
    my ( $self , $loop ) = @_;
    # learn メソッド内でチェックを行う

    if ( ! defined $loop ) {
        return;   # PASS
    }

    if (( ! defined $self->{old_layerwaits} ) || ( ! defined $self->{new_layerwaits} ) ) {
        &::Logging("INFO: old_layerwaits or new_layerwaits undefined");
        return;
    }
    
    # waitsの値が変化していないかチェックしてメッセージを表示する
    my $check = [];
    for my $l ( 0 .. $self->{layer_count} ) {
        for my $n ( 0 .. $self->{layer_member}->[$l] ) {
            my $oldwaitsstring = join ("" , @{$self->{old_layerwaits}->[$l]->[$n]});
            my $newwaitsstring = join ("" , @{$self->{new_layerwaits}->[$l]->[$n]});
            if ( $oldwaitsstring eq $newwaitsstring ) {
                push( @{$check->[$l]} , 1);  # 一致
	    } else {
                push( @{$check->[$l]} , 0);  # 不一致
	    }
        }
    }


    for my $l ( 0 .. $self->{layer_count} ) {
        my $checkstring = join ("" , @{$check->[$l]}); 
	my $fillstring = join ("", @{$self->{fillARRAY}->[$l]});

    &::Logging("DEBUG: l: $l  checkstring: $checkstring ");

        if ($checkstring eq $fillstring) {
	    # レイヤー毎にチェックする
            &::Logging("DEBUG: loop: $loop waits no change!!! layer $l ");
	    } 
    }
    return;
}

sub optimaizer {
    my ($self , $l , $n , $w , $sample ) = @_;
    # layer_initでoptimaizeerが指定されていると、ここで処理を加える。
    # waitsとbiasを更新する
    # learnメソッドのループ内で利用する

    my $debug = $self->{debug_flg}; # 0:off 1:on
    my $learn_rate = $self->{layer}->[$l]->[$n]->learn_rate();

    if ((( ! defined $self->{initdata}->{optimaizer} ) || ( $self->{initdata}->{optimaizer} eq 'None' )) && (defined $w )) {
    # optimaizerの指定がないまたは’None'の場合waitの更新

        my $tmp = $self->{old_layerwaits}->[$l]->[$n]->[$w] - ( $learn_rate * $self->{backprobacation}->[$l]->[$n]->[$w]->{first} * $self->{backprobacation}->[$l]->[$n]->[$w]->{second} * $self->{backprobacation}->[$l]->[$n]->[$w]->{third} ); 

        return $tmp;
    } elsif (( ! defined $w ) && ( $self->{initdata}->{optimaizer} eq 'None' ))  {
    # $w が指定されないとbiasの更新
    #
			# biasの更新 (ここはノード単位で1回) 後段のwaitsの更新が済んでいることが前提
			if ($l == $self->{layer_count} ) {
			    #　出力層
			    my $iota = $self->{act_funcs}->{$self->{layer_act_func}->[$l]}->( $self , $l , $n );
			    my $bias = $self->{layer}->[$l]->[$n]->bias();
			    #my $tmp = $bias - ( $learn_rate *  ($out->[$l]->[$n] - $sample->{class}->[$n]) * $iota * $out->[$l-1]->[$n]); # biasが前段の入力を得ているのがおかしいかもしれない
			    #&::Logging("DEBUG: sample found!") if defined $sample ;
			    my $tmp = $bias - ( $learn_rate *  ($self->{calc_multi_out}->[$l]->[$n] - $sample->{class}->[$n]) * $iota * 1); 
			    $self->{layer}->[$l]->[$n]->bias($tmp);
			    $self->{new_layerbias}->[$l]->[$n] = $tmp;
			    &::Logging("DEBUG: OUTLAYER learn_rate: $learn_rate bias: $bias iota: $iota new_bias: $tmp ") if $debug == 1;
                            
                            undef $tmp;
			    undef $bias;
			    undef $iota;
		        } elsif ( $l < $self->{layer_count} ) {
			    # 中間層
			    my $bias = $self->{layer}->[$l]->[$n]->bias();
			    say "l: $l n: $n back_n: $self->{layer_member}->[$l+1] " if $debug == 1;
			    my $theta = undef;
                            my $iota = undef;
                            for my $node ( 0 .. $self->{layer_member}->[$l+1] ) {
				    # 一つ後ろの層のwaitsの添字が現在のノード番号になる
                                $theta += $self->{backprobacation}->[$l+1]->[$node]->[$n]->{first}; 
				$iota += $self->{backprobacation}->[$l+1]->[$node]->[$n]->{second};
                            }
			    my $tmp = $bias - ( $learn_rate * $theta * $iota * 1 ); # biasなので重み付けは1
                            $self->{layer}->[$l]->[$n]->bias($tmp);
			    $self->{new_layerbias}->[$l]->[$n] = $tmp;
			    &::Logging("DEBUG: MIDDLELAYER learn_rate: $learn_rate bias: $bias iota: $iota theta: $theta new_bias: $tmp ") if $debug == 1;

			    undef $tmp;
			    undef $bias;
			}

    } elsif (( $self->{initdata}->{optimaizer} eq 'adam' ) &&( defined $w )) {
    # adamによるwaitsの調整 wait値を戻す
	my $grad = $self->{backprobacation}->[$l]->[$n]->[$w]->{first} * $self->{backprobacation}->[$l]->[$n]->[$w]->{second} * $self->{backprobacation}->[$l]->[$n]->[$w]->{third};
	my $mini_num = $self->{adam_params}->{mini_num};
	my $beta1 = $self->{adam_params}->{moment_beta};
	my $beta2 = $self->{adam_params}->{rms_beta};

	# 出力層では以前の値がないので0を代入する
	my $v_old = $self->{adam_waits}->[$l]->[$n]->[$w]->{v};
	   $v_old = 0 if ! defined $v_old;  # 未定義なら0
	my $s_old = $self->{adam_waits}->[$l]->[$n]->[$w]->{s};
	   $s_old = 0 if ! defined $s_old;
        my $v = $beta1 * $v_old + ( 1 - $beta1 ) * $grad;
        my $s = $beta2 * $s_old + ( 1 - $beta2 ) * $grad ** 2;
	    
	# 値を記録しておく 書き換えられる
        $self->{adam_waits}->[$l]->[$n]->[$w]->{v} = $v;
	$self->{adam_waits}->[$l]->[$n]->[$w]->{s} = $s;

	my $tmp = $self->{old_layerwaits}->[$l]->[$n]->[$w] - ( $learn_rate * ( $v / sqrt ( $s + $mini_num ) ));

	&::Logging("DEBUG: v_old: $v_old v: $v s_old: $s_old s: $s ") if $debug == 1;

	undef $mini_num;
	undef $beta1;
	undef $beta2;
	undef $v_old;
	undef $v;
	undef $s_old;
	undef $s;
	undef $grad;

	return $tmp;

    } elsif (( ! defined $w ) && ( $self->{initdata}->{optimaizer} eq 'adam' )) {
    # adamによるbiasの調整

    # biasの更新 (ここはノード単位で1回) 後段のwaitsの更新が済んでいることが前提
        if ($l == $self->{layer_count} ) {
	    #　出力層
	    my $iota = $self->{act_funcs}->{$self->{layer_act_func}->[$l]}->( $self , $l , $n );
	    my $bias = $self->{layer}->[$l]->[$n]->bias();

	    my $grad = ($self->{calc_multi_out}->[$l]->[$n] - $sample->{class}->[$n] ) * $iota * 1;
	    my $mini_num = $self->{adam_params}->{mini_num};
	    my $beta1 = $self->{adam_params}->{moment_beta};
	    my $beta2 = $self->{adam_params}->{rms_beta};

	    my $v_old = $self->{adam_bias}->[$l]->[$n]->{v};
	       $v_old = 0 if ! defined $v_old; 
	    my $s_old = $self->{adam_bias}->[$l]->[$n]->{s};
	       $s_old = 0 if ! defined $s_old;
            my $v = $beta1 * $v_old + ( 1 - $beta1 ) * $grad;
            my $s = $beta2 * $s_old + ( 1 - $beta2 ) * $grad ** 2;
	    
	    # 値を記録しておく 書き換える
            $self->{adam_bias}->[$l]->[$n]->{v} = $v;
	    $self->{adam_bias}->[$l]->[$n]->{s} = $s;

	    my $tmp = $bias - ( $learn_rate * ( $v / sqrt ( $s + $mini_num ) ));

	    $self->{layer}->[$l]->[$n]->bias($tmp);
	    $self->{new_layerbias}->[$l]->[$n] = $tmp;
	    &::Logging("DEBUG: OUTLAYER learn_rate: $learn_rate bias: $bias iota: $iota new_bias: $tmp v_old: $v_old v: $v s_old: $s_old s: $s ") if $debug == 1;
                            
            undef $tmp;
	    undef $bias;
	    undef $iota;

	    undef $mini_num;
	    undef $beta1;
	    undef $beta2;
	    undef $v_old;
	    undef $v;
	    undef $s_old;
	    undef $s;
	    undef $grad;

	} elsif ( $l < $self->{layer_count} ) {
	    # 中間層
	    my $bias = $self->{layer}->[$l]->[$n]->bias();
	    say "l: $l n: $n back_n: $self->{layer_member}->[$l+1] " if $debug == 1;
	    my $theta = undef;
            my $iota = undef;
            for my $node ( 0 .. $self->{layer_member}->[$l+1] ) {
	        # 一つ後ろの層のwaitsの添字が現在のノード番号になる
                $theta += $self->{backprobacation}->[$l+1]->[$node]->[$n]->{first}; 
		$iota += $self->{backprobacation}->[$l+1]->[$node]->[$n]->{second};
            }

	    my $grad = $theta * $iota * 1;   # biasなので1
	    my $mini_num = $self->{adam_params}->{mini_num};
	    my $beta1 = $self->{adam_params}->{moment_beta};
	    my $beta2 = $self->{adam_params}->{rms_beta};

	    my $v_old = $self->{adam_bias}->[$l]->[$n]->{v};
	    my $s_old = $self->{adam_bias}->[$l]->[$n]->{s};
            my $v = $beta1 * $v_old + ( 1 - $beta1 ) * $grad;
            my $s = $beta2 * $s_old + ( 1 - $beta2 ) * $grad ** 2;
	    # 値を記録しておく 書き換える
            $self->{adam_bias}->[$l]->[$n]->{v} = $v;
	    $self->{adam_bias}->[$l]->[$n]->{s} = $s;

	    my $tmp = $bias - ( $learn_rate * ( $v / sqrt ( $s + $mini_num ) ));

            $self->{layer}->[$l]->[$n]->bias($tmp);
	    $self->{new_layerbias}->[$l]->[$n] = $tmp;
	    &::Logging("DEBUG: MIDDLELAYER learn_rate: $learn_rate bias: $bias iota: $iota theta: $theta new_bias: $tmp v_old: $v_old v: $v s_old: $s_old s: $s ") if $debug == 1;

	    undef $tmp;
	    undef $bias;

	    undef $mini_num;
	    undef $beta1;
	    undef $beta2;
	    undef $v_old;
	    undef $v;
	    undef $s_old;
	    undef $s;
	    undef $grad;
	}

    } # elsif $w

    return;
}

sub stat {
    my $self = shift;
    # 手動処理に切り返るケースに利用する

    if (@_) {
        $self->{stat} = $_[0];
    }
    return $self->{stat};
}

sub input {
    my $self = shift;
    # dataがセットされるのみパーセプトロンに振り分けは行わない
    # calc_Multiの前に行うもの
    if (@_) {
        if ( $_[0] =~ /ARRAY/ ) {

            $self->{input} = $_[0];

	} else {
            croak "input not ARRAY!";
	}
    }

}

sub calc_multi {
    my $self = shift;
    # 引数が無ければ、学習済みでの処理
    # 引数に”learn"が在ると学習モード
    # $out(全ノード出力)を全部出力する
    # 結果だけなら $out->[-1]が1次元配列で結果となる

    my $debug = 0; # on: 1 off: 0
    
    if (@_) {
        if ( $_[0] eq 'learn' ) {
            &::Logging("DEBUG: learn mode calc_multi") if $debug == 1;
        } else {
            croak "input error";
        } 
    } else {
    # 引数なし
        if ( $self->{stat} ne 'learned' ) {
            croak "No learn perceptron...";
        }
    }

    my $out = []; #3階層に成るように各ノードの出力結果  $out->[レイヤー]->[ノード]
    my @layer = @{$self->{layer}};   # layer内はPerceptronなので、メソッドと変数をを間違えないように

    for (my $l=0; $l<=$#layer; $l++) {        
        my @nodes = @{$layer[$l]};
            for ( my $n=0; $n<=$#nodes; $n++) {
                my $l_input = [];  # 一つ前のレイヤーアウトプット レイヤー毎の入力
	        if ( $l == 0){
	        # 入力レイアー
                    $layer[$l]->[$n]->input($self->{input});
		    #my $res = $layer[$l]->[$n]->calcReLU();
		    my $method = $self->{layer_act_func}->[$l]; # layer_initの式から、scalarに展開して
		    my $res = $layer[$l]->[$n]->calcSum->${\$method};   # リファレンスを経由するとメソッドを変数で指定出来る
		    #my $res = $layer[$l]->[$n]->calcSum->ReLU();  
		    push(@{$out->[$l]} , $res);
		    undef $res;
                } elsif ( $l <= $self->{layer_count}) {
                #　中間レイアー & 出力レイヤー
                    # 前段の出力を集計 
                    for my $node ( 0 .. $self->{layer_member}->[$l-1]) {   # $nだとややこしいのであえて書き方を変える
                        push(@{$l_input} , $out->[$l-1]->[$node]);
		    }
		    $layer[$l]->[$n]->input($l_input);
		    #my $res = $layer[$l]->[$n]->calcReLU();
		    my $method = $self->{layer_act_func}->[$l];
		    my $res = $layer[$l]->[$n]->calcSum->${\$method};
		    #my $res = $layer[$l]->[$n]->calcSum->ReLU();
                    push(@{$out->[$l]} , $res);
		    undef $res;
	        }

=pod
		# メソッドを変数展開出来たので、場合分けが不要になった
	        } elsif ( $l == $self->{layer_count}) {
	        # 出力レイアー
                    for my $node ( 0 .. $self->{layer_member}->[$l-1]) {
                        push(@{$l_input} , $out->[$l-1]->[$node]);
		    }
                    $layer[$l]->[$n]->input($l_input);
		    #my $res = $layer[$l]->[$n]->calcStep();
                    my $res = $layer[$l]->[$n]->calcSum->Step();
                    push(@{$out->[$l]} , $res);
		    undef $res;
	        }
=cut
		undef $l_input;
	  } # for $n
	  undef @nodes;
    } # for $l
    undef @layer;

    $self->{calc_multi_out} = $out;

    # ARRAY ref
    return $out;
}

sub datalog_name {
    my $self = shift;
    # DatalogモジュールをMultilayerぁら使うためのメソッド

    if (@_) {
        if ($_[0] eq "" ) {
            croak "no input Error!";
	} else {
            $self->{datalog_name} = $_[0];
	}

    } else {
        croak "no input Error!";
    }
}

sub datalog_init {
    my $self = shift;
    # DatalogモジュールをMultilayerぁら使うためのメソッド

    if ( defined $self->{datalog_name} ) {
        $self->{datalog} = Datalog->new($self->{datalog_name});
    } else {
        $self->{datalog} = Datalog->new;
    }
}

sub datalog_transaction {
    my $self = shift;

    if (@_) {
        if ( $_[0] eq 'off' ) {
            $self->{datalog_transaction} = 'off';
	} elsif ( $_[0] eq 'on' ) {
            $self->{datalog_transaction} = 'on';
	} else {
            croak "input error";
	}
    } else {
        croak "input error";
    }
}

sub datalog_snapshot {
    my $self = shift;
    # dump_structureを利用してスナップショットを記録する
    # 前提はdatalog_initがされていること

    if ( ! defined $self->{datalog} ) {
        croak "no setup Datalog!!!";
    }

    my $start_waits = $self->dump_structure('check');
    my $start_waits_strings = Dumper $start_waits;
    $self->{datalog}->addlog($start_waits_strings);
    undef $start_waits;
    undef $start_waits_strings;
}

sub DESTROY {
    my $self = shift;
    # exitで終了する場合に対処する
    $self->{datalog}->commit() if $self->{datalog_transaction} eq 'on';
}

1;
