package Multilayer;
# 多層パーセプトロン用モジュール といっても低層レベルの
# Perceptronモジュールを集約して、
# 誤差逆伝搬法（バックプロパゲーション）を試す
# 学習には全体を見渡さないと出来ないので。。。
# 最終的にはネットワーク経由でノードを分散させることも考える
# ノード側PerceptronモジュールのcalcReLU,calcStepメソッドを呼び出して、計算し、
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
       $self->{initdata} = undef;

       $self->{stat} = ""; # モジュールのステータス
                           # layer_inited : 初期化済  ->learn()　が動作する
                           # learned :　学習済   ->calc_multi()が動作する
			   #
       $self->{input} = undef ;
       $self->{learn_limit} = 10000;   # 学習データ1個に対して、waitsの更新を制限する。しかし、waitsに変化がないと抜けるのでlimitになっていない
       $self->{learn_finish} = {};  #学習が終わるためのチェックリスト　ハッシュでclassラベルをチェックする

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
	if ( exists $self->{initdata}->{learn_rate} ) {
            $self->{learn_rate} = $self->{initdata}->{learn_rate};
	}

    } else {
        croak "initdata error!";
    }

    $self->{layer} = []; # 初期化　ループに対応

    for (my $l=0;$l<=$self->{layer_count};$l++) {
        my $nodes = [];
        for (my $n=0; $n<=$self->{layer_member}->[$l]; $n++) {
            push( @{$nodes} , Perceptron->new );
	    # waitsの初期化
	    if ( $l == 0 ) {
		# He初期化のためレイヤーのノード数を送る
		my $node_count = $self->{layer_member}->[$l] + 1;
		$nodes->[$n]->waitsinit($self->{input_count} , $node_count);
		#$nodes->[$n]->waitsinit($self->{input_count});  # 乱数初期化
	    } else {
		my $node_count = $self->{layer_member}->[$l] + 1;
		$nodes->[$n]->waitsinit($self->{layer_member}->[$l-1] , $node_count);  # 一つ前の階層のノード数
		#$nodes->[$n]->waitsinit($self->{layer_member}->[$l-1]);  #乱数初期化 
	    }
	    if ( defined $self->{learn_rate} ) {
		# 学習率が指定されていれば変更する
                $nodes->[$n]->learn_rate($self->{learn_rate});
	    }
        }
    push (@{$self->{layer}} , $nodes);
    undef $nodes;
    }

    $self->{stat} = "layer_inited";
}

sub takeover {
    my $self = shift;
    # dump_structureからwaitsとbiasを引き継ぐ
    # dump hashをそのまま入力されることを想定
    # 前提はlayer_initが済んでいること

    if (@_) {
        if ($_[0] =~ /HASH/) {
            if ( exists $_[0]->{biasdump} and $_[0]->{waitsdump} ) {
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
	    $nodes[$n]->waits($self->{takeover}->{waitsdump}->{$l}->{$n});
	    $nodes[$n]->bias($self->{takeover}->{biasdump}->{$l}->{$n});
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
		    waitsdump => $waitsdump,
		    biasdump => $biasdump,
                    learn_rate => $learn_rate_dump,
                   };

    if ( $self->{dump_check} == 1 ) {

        return $dumpdata;

    } else {

    open (my $fh , '> ./dump_structure.txt');

        say $fh "#dump data structure";
        say $fh "# hash data key 'layer_init' 'waitsdump' 'biasdump' 'learn_rate'";
        say $fh "# layerdata->{layer}->{node} ";
        say $fh "# waitsdump: waits data to node refARRAY";
        say $fh "# layer_init: layer_init set data";
        say $fh "# biasdump: need set for perceptron";
        say $fh "# learn_rate: extra data";

        print $fh Dumper($dumpdata);

        close($fh);
    }

    undef $dumpdata;
    undef $dt;
    undef $waitsdump;
    undef $biasdump;
    undef $learn_rate_dump;
}

sub learn {
    my $self = shift;
    # perceptronと同じフォーマットでサンプルデータを受け入れる。
    # ただし、論理は0 or 1に変更される。
    # サンプルクラス毎に最低1回収束するところで学習済みとフラグを立てる。
    # 振り分けに失敗する場合は、外部からサンプルを替えて学習を続けることで収束も可能になる

    if (! $self->{stat} eq "layer_inited") {
        croak "Still init ...";
    }

    if (!@_ ) {
        croak "no larning data";
    }

    if (@_) {
        if ($_[0] =~ /ARRAY/) {
            $self->{learn_input} = $_[0];  # learn内のみ変数
	} else {
            croak "input data format not match!";
	}
    } 

    my $debug = 0; # 0: off 1: on
    my $hand = 0; # 0: off 1: on  手動実行時にほしい表示 収束するか傾向を見る場合

    $self->datalog_init(); 

    $self->{datalog}->begin_work() if $self->{datalog_transaction} eq 'on';

    # チェック比較時に利用する配列
    my $fillARRAY = [];
    for my $l ( 0 .. $self->{layer_count} ){
        for my $n ( 0 .. $self->{layer_member}->[$l] ) {
            push(@{$fillARRAY->[$l]} , 1); 
        }
    }

    my $loop = 0;
    # 入力して各層を計算していく
    for my $sample (@{$self->{learn_input}}) {
        $loop++;

        my $sample_flg = 1;
	my $sample_count = 0;
        while ( $sample_flg ) {  

            if ($sample_count >= $self->{learn_limit} ) {
                &::Logging("learn limit over!");
		$sample_flg = 0;
		exit;
	    }
            $sample_count++;

            &::Logging("Loop: $loop start ------------------") if $debug == 1;

	    #my $out = []; #3階層に成るように各ノードの出力結果  $out->[レイヤー]->[ノード]

	    $self->{tmp} = [];
	    @{$self->{tmp}} = @{$sample->{input}}; # デリファレンスでリークを回避

            $self->input($self->{tmp});
	    my $out = $self->calc_multi('learn');

	    undef $self->{tmp};

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
                #重み付けの更新 

=pod
	        #2乗誤差関数  (Sigma(出力層 - sample_class)^2 )/2  理論的話だったので計算は不要だった。
		my $esum = 0;
		for my $node ( 0 .. $self->{layer_member}->[-1]) {
		    $esum += ($out->[$self->{layer_count}]->[$node] - $sample->{class}->[$node] ) ^ 2;
		}
		$esum = $esum / 2;
		&::Logging("DEBUG: 誤差関数 $esum");
=cut

                # 現在のwaitsを取得する biasを除く
		my $old_layerwaits = [];  # 3次元配列
		for my $l ( 0 .. $self->{layer_count}) {
                    for my $node ( 0 .. $self->{layer_member}->[$l]) {
                        my $waits_old = $self->{layer}->[$l]->[$node]->waits();
                        push(@{$old_layerwaits->[$l]} , $waits_old ); #waitsはノード単位
		    }	
                } 

		&::Logging("DEBUG: old_layerwaits") if $debug == 1;
		print Dumper $old_layerwaits if $debug == 1;

		my $new_layerwaits = [];
		my $new_layerbias = [];
		my $backprobacation = [];
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
                                    $first += $backprobacation->[$l+1]->[$nsum]->[$n]->{first} * $backprobacation->[$l+1]->[$nsum]->[$n]->{second} * $new_layerwaits->[$l+1]->[$nsum]->[$n]; 
				}
				# 活性化関数がReLUのケース
				my $bias = $self->{layer}->[$l]->[$n]->bias(); 
				if ( $out->[$l]->[$n] >= $bias ) {
		                    $second = 1; # 活性化関数 の微分 ReLU関数
			        } else {
                                    $second = 0;
				}
				$third = $sample->{input}->[$w]; # 入力値 ########
				$backprobacation->[$l]->[$n]->[$w] = clone({ first => $first , second => $second }); # 次の回の計算で利用される
                                my $tmp = $old_layerwaits->[$l]->[$n]->[$w] - ( $learn_rate * $backprobacation->[$l+1]->[$self->{layer_member}->[$l+1]]->[$n]->{first} * $backprobacation->[$l+1]->[$self->{layer_member}->[$l+1]]->[$n]->{second} * $third );
                                push (@{$waits_delta} , $tmp); #調整したwaits

				undef $tmp;
                            } # for 入力層 w #################################
		        } else {
                        # 中間層、出力層
			    for my $w ( 0 .. $self->{layer_member}->[$l-1] ) {   # waitsは順方向ループ 前の層のノード数がwaitsの数
			        &::Logging("DEBUG: layer: $l node: $n waits: $w") if $debug == 1;

				my ( $first , $second , $third ) = ( undef , undef , undef );

                                if ($l == $self->{layer_count}) {
		                # 出力層の重み付け調整  ステップ関数なので、ここで＋ーの方向が決まる
			            $first = $out->[$l]->[$n] - $sample->{class}->[$n];  # 誤差関数の偏微分->今回の出力からクラスラベル差
				    # Step関数のケース
                                    $second = $out->[$l]->[$n];   # 活性化関数の偏微分 ->出力値そのまま
                                    $third = $out->[$l-1]->[$w];   #入力から得られた結果の偏微分 -> 前の層からの入力
				    $backprobacation->[$l]->[$n]->[$w] = clone({ first => $first , second => $second }); # 次の層の計算で利用される

				    &::Logging("DEBUG: learn_rate: $learn_rate first: $first second: $second third $third ") if $debug == 1;
                                    my $tmp = $old_layerwaits->[$l]->[$n]->[$w] - ( $learn_rate * $first * $second * $third ); 
                                    push (@{$waits_delta} , $tmp);  # 調整したwaits 

				    undef $tmp;
			        } elsif ( $l < $self->{layer_count} ) {
                                # 中間層
			        # waitsは後層すべてのノードに影響するので、合計が必要になる waitsの添字は現在のノード番号
			            for my $nsum ( 0 .. $self->{layer_member}->[$l+1] ) {
                                        $first += $backprobacation->[$l+1]->[$nsum]->[$n]->{first} * $backprobacation->[$l+1]->[$nsum]->[$n]->{second} * $new_layerwaits->[$l+1]->[$nsum]->[$n]; 
				    }
				    # ReLU関数の微分
				    my $bias = $self->{layer}->[$l]->[$n]->bias(); 
				    if ( $out->[$l]->[$n] >= $bias ) {
		                        $second = 1; # 活性化関数 
			            } else {
                                        $second = 0;
				    }
                                    $third = $out->[$l-1]->[$w];  # 前の層からの入力  ノードの番号はwaitsの添字
				    $backprobacation->[$l]->[$n]->[$w] = clone({ first => $first , second => $second }); # 次の回の計算で利用される

				    # 一つ後層の計算結果を利用する
                                    my $theta = undef;
				    my $iota = undef;
                                    for my $node ( 0 .. $self->{layer_member}->[$l+1]) {
                                       $theta += $backprobacation->[$l+1]->[$node]->[$n]->{first};   # 今の$nが後ろのwaitの添字
				       $iota += $backprobacation->[$l+1]->[$node]->[$n]->{second};
                                    }
                                    my $tmp = $old_layerwaits->[$l]->[$n]->[$w] - ( $learn_rate * $theta * $iota * $third );
                                    push (@{$waits_delta} , $tmp); #調整したwaits
                                    
                                    undef $tmp;
			       }
                            } # for $w  ############################################
		        } # if $l==0 esle

			# biasの更新 (ここはノード単位で1回)
			if ($l == $self->{layer_count} ) {
			    #　出力層
			    my $bias = $self->{layer}->[$l]->[$n]->bias();
			    my $tmp = $bias - ( $learn_rate *  ($out->[$l]->[$n] - $sample->{class}->[$n]) * $out->[$l]->[$n] * $out->[$l-1]->[$n]); 
			    $self->{layer}->[$l]->[$n]->bias($tmp);
			    $new_layerbias->[$l]->[$n] = $tmp;
                            
                            undef $tmp;
			    undef $bias;
		        } elsif ( $l <= $self->{layer_count} ) {
			    # 中間層
			    my $bias = $self->{layer}->[$l]->[$n]->bias();
			    say "l: $l n: $n back_n: $self->{layer_member}->[$l+1] " if $debug == 1;
			    my $theta = undef;
                            my $iota = undef;
                            for my $node ( 0 .. $self->{layer_member}->[$l+1] ) {
				    # 一つ後ろの層のwaitsの添字が現在のノード番号になる
                                $theta += $backprobacation->[$l+1]->[$node]->[$n]->{first}; 
				$iota += $backprobacation->[$l+1]->[$node]->[$n]->{second};
                            }
			    my $tmp = $bias - ( $learn_rate * $theta * $iota * 1 ); # biasなので重み付けは1
                            $self->{layer}->[$l]->[$n]->bias($tmp);
			    $new_layerbias->[$l]->[$n] = $tmp;

			    undef $tmp;
			    undef $bias;
			}
                        # ノードのwaitsを保持する
                        $new_layerwaits->[$l]->[$n] = clone($waits_delta); 

			&::Logging("DEBUG: backprobacation l: $l n: $n") if $debug == 1 ;
                        print Dumper $backprobacation if $debug == 1;
		    } # for $n


		} # for $l

		&::Logging("DEBUG: new_layerwaits") if $debug == 1;
		print Dumper $new_layerwaits if $debug == 1;

                # new_layerwaitsに値が入っていることを確認して、
		for my $l ( 0 .. $self->{layer_count} ) {
                    for my $n ( 0 .. $self->{layer_member}->[$l] ) {
			if ( $l == 0 ) {
			    # 入力層
                            for my $w ( 0 .. $self->{input_count}) {
                                if ( ! defined $new_layerwaits->[$l]->[$n]->[$w] ) {
                                    croak "new_layerwaits undef detected!! l: $l n: $n w: $w";
				    exit;
			        }
                            }
			} else {
			    # 中間層、出力層
                            for my $w ( 0 .. $self->{layer_member}->[$l-1] ) {
                                if ( ! defined $new_layerwaits->[$l]->[$n]->[$w] ) {
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
                        $self->{layer}->[$l]->[$n]->waits($new_layerwaits->[$l]->[$n]);
                    }
		}

		my $new_structure = { 
			              initdata => $self->{initdata},
			              waits => $new_layerwaits,
			              bias => $new_layerbias,
				      out => $out,
			            }; 
		my $new_structure_strings = Dumper $new_structure;

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

		undef $newstructure;
		undef $new_layerwaits;
		undef $new_layerbias;

		&::Logging("DEBUG: loop $loop Change waits value  ------------------------") if $hand == 1;
		&::Logging("DEBUG: loop $loop Retry! $sample_count ------------------------") if $debug == 1;

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
		
                for my $l ( 0 .. $self->{layer_count} ) {
                    my $checkstring = join ("" , @{$check->[$l]}); 
		    my $fillstring = join ("", @{$fillARRAY->[$l]});

                    if ($checkstring eq $fillstring) {
			    # レイヤー毎にチェックする
                        &::Logging("DEBUG: loop: $loop waits no change!!! layer $l sample_count: $sample_count") if $hand == 1;
			$sample_flg = 0;
		    } 
                }

            } # if sampleclassstring

	} # while 

    } # for sample 

    $self->{datalog}->commit() if $self->{datalog_transaction} eq 'on';

    &::Logging("DEBUG: learn_finish data") if $debug == 1;
    # class毎の学習が完了したか目視用のDump
    print Dumper $self->{learn_finish} if $hand == 1;

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
    undef $self->{learn_input};

    return $self->{stat};  # 完了ならlearndでなければlayer_initedが返る
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
		    my $res = $layer[$l]->[$n]->calcReLU();
		    push(@{$out->[$l]} , $res);
		    undef $res;
                } elsif ( $l < $self->{layer_count}) {
                #　中間レイアー
                    # 前段の出力を集計 
                    for my $node ( 0 .. $self->{layer_member}->[$l-1]) {   # $nだとややこしいのであえて書き方を変える
                        push(@{$l_input} , $out->[$l-1]->[$node]);
		    }
		    $layer[$l]->[$n]->input($l_input);
		    my $res = $layer[$l]->[$n]->calcReLU();
                    push(@{$out->[$l]} , $res);
		    undef $res;
	        } elsif ( $l == $self->{layer_count}) {
	        # 出力レイアー
                    for my $node ( 0 .. $self->{layer_member}->[$l-1]) {
                        push(@{$l_input} , $out->[$l-1]->[$node]);
		    }
                    $layer[$l]->[$n]->input($l_input);
                    my $res = $layer[$l]->[$n]->calcStep();
                    push(@{$out->[$l]} , $res);
		    undef $res;
	        }
		undef $l_input;
	  } # for $n
	  undef @nodes;
    } # for $l
    undef @layer;

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

sub DESTROY {
    my $self = shift;
    # exitで終了する場合に対処する
    $self->{datalog}->commit();
}

1;
