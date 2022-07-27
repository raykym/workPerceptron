#!/usr/bin/env perl
#
# simple perceptronの学習ななど、自作実装で動作を確認する。
# AND OR NAND NORについて動作することを確認
#
# 高卒でもわかる機械学習　サイトを参考に作成
#
use strict;
use warnings;
use utf8;
use feature ':5.13';

binmode 'STDOUT' , ':utf8';

use Time::HiRes qw/ time /;
use Data::Dumper;
use Devel::Size;
use Devel::Cycle;

$|=1;

srand();

sub Logging {
        my $logline = shift;
        my $dt = time();
        say "$dt | $logline";

        undef $dt;
        undef $logline;

        return;
}


my $learndata_ANDgate = [ 
	              { 
		        class => 1 ,
		        input => [ 1 , 1 ]
		      },	
		      {
		        class => -1 ,
			input => [ -1 , 1 ]
		      },
		      {
		        class => -1 ,
			input => [ 1 , -1 ]
		      },
		      {
			class => -1 ,
			input => [ -1 , -1 ]
		      },
                    ];

my $learndata_ORgate = [ { 
		        class => 1 ,
		        input => [ 1 , 1 ]
		      },	
		      {
		        class => 1 ,
			input => [ -1 , 1 ]
		      },
		      {
		        class => 1 ,
			input => [ 1 , -1 ]
		      },
		      {
			class => -1 ,
			input => [ -1 , -1 ]
		      }
                    ];

my $learndata_NANDgate = [ 
	              { 
		        class => -1 ,
		        input => [ 1 , 1 ]
		      },	
		      {
		        class => 1 ,
			input => [ -1 , 1 ]
		      },
		      {
		        class => 1 ,
			input => [ 1 , -1 ]
		      },
		      {
			class => 1 ,
			input => [ -1 , -1 ]
		      },
                    ];

my $learndata_NORgate = [ { 
		        class => -1 ,
		        input => [ 1 , 1 ]
		      },	
		      {
		        class => -1 ,
			input => [ -1 , 1 ]
		      },
		      {
		        class => -1 ,
			input => [ 1 , -1 ]
		      },
		      {
			class => 1 ,
			input => [ -1 , -1 ]
		      }
                    ];

my $learndata_XORgate = [ { 
		        class => -1 ,
		        input => [ 1 , 1 ]
		      },	
		      {
		        class => 1 ,
			input => [ -1 , 1 ]
		      },
		      {
		        class => 1 ,
			input => [ 1 , -1 ]
		      },
		      {
			class => -1 ,
			input => [ -1 , -1 ]
		      }
                    ];


=pod
		   
    # 単純パーセプトロンの学習をテストした
    # 教育データ　ANDgate ORgate NandGateを書き換えて重み付けを教育する

    #print Dumper $learndata_ANDgate;
    #say "";

    my $unit = Perceptron->new();

       $unit->learn_simple($learndata_ORgate);

    my $waits = $unit->waits();
    my $bias = $unit->bias();

    say "";
    say "learn finish value";
    print "bias: $bias\n";
    print "waits: ";
    print "$_ " for @{$waits};
    say "";

    exit;
    ###########################################



    # XORゲートを作るには、ANDゲートとNANDゲートとORゲートを学習したパーセプトロンを用意して、
    # 入出力を組み合わせれば出来るはず、機械学習で一気に作れるのかはわからないが、
    # そういうことなのではないかと、テストしてみる。

    my $andGate = Perceptron->new();
    my $orGate = Perceptron->new();
    my $nandGate = Perceptron->new();

       $andGate->learn_simple($learndata_ANDgate);
       Logging("andGate complete!");

       $orGate->learn_simple($learndata_ORgate);
       Logging("orGate compleate!");

       $nandGate->learn_simple($learndata_NANDgate);
       Logging("nandGate compleate  setup OK!");

       # XORになるのか入力して試してみる
       my $indata = [ 1 , 1 ];
       $orGate->input($indata);
       $nandGate->input($indata);  #入力は同じもの

       my $res_or = $orGate->calc();
       Logging("orGate calc OK");
       my $res_nand = $nandGate->calc();
       Logging("nandGate calc OK");

       my $result = [ $res_or, $res_nand ]; # 結果を配列に入れる

       Logging("中間層 @{$result}");

       $andGate->input($result);

       my $res = $andGate->calc();

       Logging("結果　 $res  期待値は-1");

       exit;
       ####################################

		    # multilayer用 clasはs0,1 に変更される。
		    # あえて大きな数値を入れて大体の感じでXORを表現する
		    # 2層構造のXORは最低100を入力しないと学習出来ない
		    #
     my $multi_learndata_XORgate = [ { 
		        class => [ 1 ],
		        input => [ 1000 , 1000 ]
		      },	
		      {
		        class => [ 0 ],
			input => [ 0 , 1000 ]
		      },
		      {
		        class => [ 0 ],
			input => [ 1000 , 0 ]
		      },
		      {
			class => [ 1 ],
			input => [ 0 , 0 ]
		      }
                    ];


    # ２層パーセプトロンを構成して、XOR回路を学習させる
    # 何回か動かすと何故か失敗することがある。。。？

    my $structure = { 
	              layer_member  => [ 1 , 0 ],
		      input_count => 1 ,
		      learn_rate => 0.034
	            };



    my $multilayer = Multilayer->new();
       $multilayer->layer_init($structure);

       $multilayer->disp_waits();

       $multilayer->learn($multi_learndata_XORgate);

       $multilayer->disp_waits();

       # 大きな数値を入れるとXORの動作をしている

       say "input [ 1000 , 1000 ]";
       $multilayer->input([1000 , 1000 ]);
       my $ret = $multilayer->calc_multi();
       say @{$ret->[-1]};

       say "input [ 1000 , 0 ]";
       $multilayer->input([1000 , 0 ]);
        $ret = $multilayer->calc_multi();
       say @{$ret->[-1]};

       say "input [ 0 , 1000 ]";
       $multilayer->input([1000 , 0 ]);
        $ret = $multilayer->calc_multi();
       say @{$ret->[-1]};

       # 単位マトリクスではXORの動作をしない

       say "input [ 0 , 0 ]";
       $multilayer->input([0 , 0 ]);
        $ret = $multilayer->calc_multi();
       say @{$ret->[-1]};

       say "input [ 1 , 0 ]";
       $multilayer->input([1 , 0 ]);
        $ret = $multilayer->calc_multi();
       say @{$ret->[-1]};

       say "input [ 0 , 1 ]";
       $multilayer->input([0 , 1 ]);
        $ret = $multilayer->calc_multi();
       say @{$ret->[-1]};

       say "input [ 1 , 1 ]";
       $multilayer->input([1 , 1 ]);
        $ret = $multilayer->calc_multi();
       say @{$ret->[-1]};

       my $total_size = Devel::Size::total_size($multilayer);

       Logging("DEBUG: total size: $total_size byte");

       exit;
       ###########################################
=cut

       # 4つの入力、始点、終点で直線を表すデータを与えて、角度の違うものを仕分ける
       # サンプルデータの順番で結果が不安定になる。？？？？？　識別は出来ていない、
       # ちゃんと図形を見て重ならない関数を用意して、識別を繰り返す。
       # ケースごとにFINISH出来るケースがあれば成功する。。。
       #
       # onlineということで、データを生成して学習を繰り返してみる
       # 学習を繰り返せば正解ということでもなく、
       # クラスラベルで成功した場合に、別途確認して、だめなら、
       # 再度学習を繰り返して、判定できるまで繰り返す必要が在る。
       # サンプルを全量通して、学習完了にはならないということ、
       # 過学習というケースも在るらしいので、それで収束しないケースも在るのかも。
       #
       # データ収束条件
       # 乱数100まで、一度に20個のデータを与え、学習処理が終わるのは設定範囲200以下(learn_flg)
       # 学習が完了フラグが立って、verificationループに入って、3から７０程度で収束する
       # ２層構造 layer_member = [ 1 , 1 ]の状態
       #
       # 例えば、乱数を1000まで利用するとほとんど学習に成功しなくなる。
       # レイヤーを3層に増やした場合でも同じく、学習がほぼ完了しない
       # 乱数を1000まで使う場合、1度に20個は替えずに500ループまで拡張すると、成功する確率がある。
       # 2000ループ(learn_flg)にすると収束する
       # 収束するが、dump機能により復元、チェックすると通らない。。。学習が足りていない
       # どこまでやれば良いのか？
       #
       # 3層以上で収束したことが無い。。。
       #
       # どうやら、class 01は０から20の範囲で成立する。
       # xの値が100とか1000の乱数で指定してたが、グラフをよく見ると、class10はマイナスのグラフなので、
       # 範囲に入っていなかった。
       # 20を超えるとclass10がマイナスに増大するので、大きく反応するので、10の範疇になるのだろう。。。
       # class01も大きな数字になるのだけれど。。。この違いはなんだろう？
       # ｘの範囲を-1000から1000に変更したところ収束しなくなった。

    my $structure = { 
	              layer_member  => [ 3 , 3 , 1 ],
		      input_count => 3 ,
		      learn_rate => 0.34
	            };

    my $multilayer = Multilayer->new();
       $multilayer->layer_init($structure);

       my $total_size_init = Devel::Size::total_size($multilayer);

       $multilayer->disp_waits();

       my $multi_learndata_online = undef;

 my $verification_flg = 1;
 my $verification_count = 0;
 my $ver_chk = [];

 while ( $verification_flg ) {
    $verification_count++;

    if ($verification_count >= 20000 ) {
        Logging("DEBUG: verification_count over");
	exit;
        $verification_flg = 0;
    }

       my $learn_flg = 1;  # loop: 1 not loop 0
       my $learn_count = 0;  # サンプルデータとして与えるグループ数、グループ内のデータ数はforループに依存する

       #機械学習 サンプルデータを生成して繰り返す
       while ($learn_flg) { # classラベルが最低1回ずつは学習成功するまで

         # $learn_flg = 0; # 手動処理の場合の追加

	   $multi_learndata_online = []; # ループ単位で初期化する

           $learn_count++;
	   Logging("DEBUG: learn_count: $learn_count");
	   if ($learn_count >= 20000 ) {
               Logging("DEBUG: learn_count over");
	       #exit;
               $learn_flg = 0;
	   }

           srand();
	   # データが直線の始点と終点という考えでデータを入力していた。
	   # 10個のサンプルを作成して、チェックする学習が完了しない場合はループする
	   # 学習が完了しても分類に失敗するケースが多々ある
           for my $count ( 1 .. 100 ) {
	       # y = -2(x+20)^2 + 100  
               my $x1 = 1000 - int(rand(2000));
               my $x2 = 1000 - int(rand(2000));
               my $y1 = -2 * ($x1 + 20)^2 + 100;
               my $y2 = -2 * ($x2 + 20)^2 + 100;
	       my $learndata_a = {};
	          $learndata_a->{class} = [ 1 , 0 ];
	          $learndata_a->{input} = [ $x1 , $y1 , $x2 , $y2 ];
	       push(@{$multi_learndata_online} , $learndata_a);


	       # y = (x-20)^2 +100
                $x1 = 1000 - int(rand(2000));
                $x2 = 1000 - int(rand(2000));
                $y1 = ($x1 - 20)^2 - 100;
                $y2 = ($x2 - 20)^2 - 100 ;
	       my  $learndata_b = {};
	          $learndata_b->{class} = [ 0 , 1 ];
	          $learndata_b->{input} = [ $x1 , $y1 , $x2 , $y2 ];
	       push(@{$multi_learndata_online} , $learndata_b);
           }

           my $stat = $multilayer->learn($multi_learndata_online);

	   if ($stat eq 'learned') {
               $learn_flg = 0;
	       # ここにwhileの下の処理を入れてチェックすればverificationのループは不要だが、思考過程がわかるようにこのままに
	       Logging("learn finish!!!");
	   } else {
               Logging("learn not yet!");
	   }
       } # while

       # class毎に成功が出て、学習終了しても、適切に学習出来てないケースが在る
       # そのため、verificationループでラップして、実際に成功するまでループさせることに

       my $x1 = 1000 - int(rand(2000));
       my $x2 = 1000 - int(rand(2000));
       my $y1 = -2 * ($x1 + 20)^2 + 100;
       my $y2 = -2 * ($x2 + 20)^2 + 100 ;
       Logging("DEBUG: x1: $x1 y1: $y1 x2: $x2 y2: $y2");

       $multilayer->input([ $x1 , $y1 , $x2 , $y2 ]); 
       my  $ret = $multilayer->calc_multi();
       say "!!!! CHECK: @{$ret->[-1]}  hope: 10 -----";    # 1 ,0 を期待

       my $retstring = join ("" , @{$ret->[-1]});
       push(@{$ver_chk} , 'ok' ) if $retstring eq '10';

        $x1 = 1000 - int(rand(2000));
        $x2 = 1000 - int(rand(2000));
        $y1 = ($x1 - 20)^2 - 100;
        $y2 = ($x2 - 20)^2 - 100;
       Logging("DEBUG: x1: $x1 y1: $y1 x2: $x2 y2: $y2");

       $multilayer->input([ $x1 , $y1 , $x2 , $y2 ]); 
       $ret = $multilayer->calc_multi();
       say "!!!! CHECK: @{$ret->[-1]} hope: 01 -----";    # 0 , 1 を期待

        $retstring = join ("" , @{$ret->[-1]});
       push(@{$ver_chk} , 'ok' ) if $retstring eq '01';

       Logging("DEBUG: verification: $verification_count ");

       # classラベル毎にチェックして"okok"であれば完了
       my $ver_chk_string = join ("" , @{$ver_chk});
       if ( $ver_chk_string eq 'okok' ) {
	   # verification ループを止める
           $verification_flg = 0;	       

           print Dumper $structure;
           $multilayer->disp_waits();
       } else {
	   $ver_chk = [];
           $learn_flg = 1; # ループを差し戻す
       }


 } # while verification

       Logging("DEBUG: init time total size: $total_size_init byte");

       my $total_size = Devel::Size::total_size($multilayer);
       Logging("DEBUG: total size: $total_size byte");


       $multilayer->dump_structure();

       #################################################
       exit;

=pod

    # dump_structureを用意したので、読み込みを試す。
    # 試してみたところ、困ったことに、復元は出来ていたが、
    # 学習が不完全で、確率的に成功しているだけということが分かった。。。。
    # 過学習どころか不完全だったことが分かった。。。。

    my $dumpData = require './dump_structure.txt';

    my $multilayer = Multilayer->new();
       my $total_size = Devel::Size::total_size($multilayer);
       Logging("DEBUG: total size: $total_size byte");

       $multilayer->layer_init($dumpData->{layer_init});
        $total_size = Devel::Size::total_size($multilayer);
       Logging("DEBUG: total size: $total_size byte");

       $multilayer->takeover($dumpData);
        $total_size = Devel::Size::total_size($multilayer);
       Logging("DEBUG: total size: $total_size byte");

       # dumpしただけで、learnedになっていないので、
       $multilayer->stat('learned');

       my $x1 = int(rand(100));
       my $x2 = int(rand(100));
       my $y1 = -2 * ($x1 + 20)^2 + 100;
       my $y2 = -2 * ($x2 + 20)^2 + 100 ;
       Logging("DEBUG: x1: $x1 y1: $y1 x2: $x2 y2: $y2");

       $multilayer->input([ $x1 , $y1 , $x2 , $y2 ]); 
       my  $ret = $multilayer->calc_multi();
       say "!!!! CHECK: @{$ret->[-1]}  hope: 10 -----";    # 1 ,0 を期待

        $x1 = int(rand(100));
        $x2 = int(rand(100));
        $y1 = ($x1 - 20)^2 - 100;
        $y2 = ($x2 - 20)^2 - 100;
       Logging("DEBUG: x1: $x1 y1: $y1 x2: $x2 y2: $y2");

       $multilayer->input([ $x1 , $y1 , $x2 , $y2 ]); 
       $ret = $multilayer->calc_multi();
       say "!!!! CHECK: @{$ret->[-1]} hope: 01 -----";    # 0 , 1 を期待

       exit;
       ################################
=cut

package Perceptron;
# シンプルなパーセプトロンのモデルを作ってみる

use Carp;
use FindBin;
use lib "$FindBin::Bin/lib";
use SPVM 'Util';
use Clone qw/clone/;
use Scalar::Util qw/ weaken /;

srand(); # 個別に乱数表が用いられる

sub new {
    my $proto = shift;
    my $class = ref $proto || $proto;

    my $self = {};
       $self->{bias} =0;
       $self->{waits} = "";
       $self->{input} = "";
       $self->{stat} = "init";
       $self->{learn_rate} = 0.33;
       $self->{learn_input} = "";
       $self->{limit} = 500;

       bless $self , $class;

    return $self;
}

sub bias {
    my $self = shift;
    # 乱数を自動入力される想定

    if (@_) {
           $self->{bias} = $_[0];
    }

    return $self->{bias};
}

sub waits {
    my $self = shift;
    # waitの入力を一括で行う場合
    # biasを含まない想定
    #
    my @waits = ();

    if (@_) {
        if ( $_[0] =~ /ARRAY/ ) {
            $self->{waits} = $_[0];
            return;
	} else {
            @waits = @_;
	    $self->{waits} = \@waits;
	    return; 
	}
    } # if @_

    return $self->{waits};
}

sub input {
    my $self = shift;

    # レファレンス入力を想定

    my @input = ();
    
    if (@_) {
        if ( ! defined $_[1] ) {
        # 2つ目の値がない
	    if ($_[0] =~ /ARRAY/) {
            # 入力がリファレンスのみ
	        $self->{input} = $_[0];    
		return;
	    } else {
                croak "input refarence not ARRAY";
	    }
	} else {
	    #引数が複数存在する
	    @input = @_;
            $self->{input} = \@input;
	}
    }

    return $self->{input};
}

sub calc {
    my $self = shift;

    my @waits = @{$self->{waits}};
    my @input = @{$self->{input}};

    if ($#waits != $#input ) {
        croak "wait input miss match!";
	exit;
    }

    my $sum = 0;
    # inputが270以下ならこのままで良い
    #for (my $i=0; $i<=$#input ; $i++) {
    #    $sum += ($input[$i] * $waits[$i]);
    #} 

    $sum = SPVM::Util->onedinnersum($self->{input} , $self->{waits});

    if ( $sum >= $self->{bias} ){
        return 1;
    } elsif ( $sum < $self->{bias} ) {
        return -1;
    }
}

sub calcReLU {
    my $self = shift;
 
    my @waits = @{$self->{waits}};
    my @input = @{$self->{input}};

    if ($#waits != $#input ) {
        croak "wait input miss match!  $#waits | $#input ";
	exit;
    }

    my $sum = 0;

    $sum = SPVM::Util->onedinnersum($self->{input} , $self->{waits});

    # ReLU関数
    if ( $sum >= $self->{bias} ){
        return $sum;
    } elsif ( $sum < $self->{bias} ) {
        return 0;
    }
}

sub calcStep {
    my $self = shift;

    my @waits = @{$self->{waits}};
    my @input = @{$self->{input}};

    if ($#waits != $#input ) {
        croak "wait input miss match!  $#waits | $#input";
	exit;
    }

    my $sum = 0;

    $sum = SPVM::Util->onedinnersum($self->{input} , $self->{waits});

    # step関数
    if ( $sum >= $self->{bias} ){
        return 1;
    } elsif ( $sum < $self->{bias} ) {
        return 0;
    }
}

sub waitsinit {
    my $self = shift;

    # waitを乱数で初期化する
    #  引数に入力数があれば、その数でwaitsを生成する。
    # ・inputが先に決まっている場合、オンライン教師あり学習の想定で利用する
    # ・learn_simpleメソッドから呼ばれるケース
    # MultilayerではReLU関数なので、乱数をHe初期化に変更
    
    my @waits = ();
    if (@_) {
	# 引数がある場合、Multilayerからの呼び出し
        if ($_[0] =~ /\d?/) {
            my $cnt = $_[0];
	    my $node_count = $_[1];  # He初期化用

	    for (my $i=0; $i<=$cnt; $i++) {
		# He初期化パラメータがある場合
		if ( defined $node_count ) {
                    my $rand = rand( 2 / $node_count );
	            push(@waits , $rand);
		} else {
	            my $rand = rand(1);
	            push(@waits , $rand);
	        }
	    }
            $self->{waits} = \@waits;

            my $rand = rand(1);
            $self->bias($rand);

            return;
        }
    }

    if ( $self->{input} =~ /ARRAY/) {
        my @tmp = @{$self->{input}};
        my $cnt = $#tmp;

	for (my $i=0; $i<=$cnt; $i++) {
            my $rand = rand(1);
	    push(@waits , $rand);
	}
        $self->{waits} = \@waits;

	undef @tmp;

    } elsif ( $self->{learn_input} =~ /ARRAY/) {
	# learn_sinpleメソッド 
	my @tmp_in = @{$self->{learn_input}->[0]->{input}};
	   
	my $flg = 1;
	while ($flg){
	    for (my $i=0;$i<=$#tmp_in; $i++) {
                my $rand = rand(1); 
                push(@waits, $rand);
	    }
            if ( $waits[0] != $waits[1] ) {
                $flg = 0;
	    }
        }

	$self->{waits} = \@waits;

        my $rand = rand(1);
        $self->bias($rand);

	undef @tmp_in;

	return;   
    } else {
        croak "Error! no input!";
    }
}

sub learn_rate {
    my $self = shift;
    # 学習率の設定、読み出し

    if (@_) {
        if ($_[0] >= 1) {
            croak "learn rate over";
	} else {
            $self->{learn_rate} = $_[0];
	    return $self->{learn_rate};
	}

    } else {
        return $self->{learn_rate};
    }
}

sub learn_simple {
    my $self = shift;
    # 教育用データ
    # 2次元配列を読み込んで学習する。

    my $debug = 0;   # 表示 0:off 1:on

   my @learn_input = @{$_[0]};
   $self->{learn_input} = \@learn_input;     
    
   $self->waitsinit();   
   #my $rand = rand(1);  # waitsinitへ移動
   #$self->bias($rand);

   my $loop = 0; # sample数
   my $before_sum = undef; # 初期化

   # sampleを一つづつ読み出して学習する
   for my $sample (@learn_input) {
       $loop++;
       my $sample_flg = 1;
       my $sample_cnt = 0; # limitカウント

   # limit値まで繰り返して学習を行う
   while ($sample_flg) { 
	  $sample_cnt++;

	  if ($sample_cnt >= $self->{limit} ) {
	      croak "DEBUG intterapt $self->{limit} count";
	  }

       &::Logging("-------------") if $debug == 1;
       &::Logging("loop: $loop sample cnt $sample_cnt --------") if $debug == 1;

       if ( $debug == 1 ) {
           print "waits :";
           print "$_ " for @{$self->{waits}};
           say "";
           print "sample :";
           print "$_ " for @{$sample->{input}};
           say "";
           say "bias: $self->{bias}";
       }

       my $sum = SPVM::Util->onedinnersum($sample->{input} , $self->{waits});   
=pod
       # inputが270以下ならこのままで良い
       my $sum=0;
       my @input = @{$sample->{input}};
       my @waits = @{$self->{waits}};
       for (my $i=0; $i<=$#input ; $i++) {
           $sum += ($input[$i] * $waits[$i]);
       } 
=cut

       my $result = 0;
       if ( $sum >= $self->{bias} ) {
	       $result = 1;  # classlabel判定
       } elsif ( $sum < $self->{bias} ) {
	       $result = -1;
       }

       #誤差関数部分   E = max(0 , -twx);
       #
       if ( $sample->{class} eq $result ) {
           # サンプルクラスとリザルトが一致していればパス
	   $sample_flg = 0;   # max 0の部分
	   &::Logging("sum: $sum bias: $self->{bias}") if $debug == 1;
	   &::Logging("loop $loop sample cnt $sample_cnt FINISH!") if $debug == 1;

       } else {
           # サンプルと結果が合わないので調整する  max -twxの部分
	   &::Logging("Adjustment sum: $sum bias: $self->{bias} classlabel: $sample->{class}") if $debug == 1;

          my $theta = $self->learn_rate();
	  my $delta = undef;
          my $error_rate = $self->{bias} - $sum;
	  &::Logging("ERROR RATE: $error_rate") if $debug == 1;
          # error_rateが正の場合 $sample->{class}は1
	  # error_rateが負の場合 $sample->{class}は-1

             $delta = -1 * $theta * $error_rate;   # -twxに該当

	  if ($delta == 0) {
              croak "DEBUG: delta 0....";
	  }

	  if (abs($error_rate) >= 500 ) {
              croak "DEBUG: Error rate 500 over";
	  }

          # 更新処理
          $self->{bias} += ($delta / $self->{bias});

	  &::Logging("delta: $delta sum: $sum theta: $theta") if $debug == 1;

	  #my @waits = map {$_ += ($delta / $_) } @{$self->{waits}};
	  my $SPVM_waits = SPVM::Util->map_learnsimple($delta , $self->{waits});
	  my $tmp = $SPVM_waits->to_elems();
	  my @waits = @{$tmp};
	  undef $SPVM_waits;

	  $self->waits(@waits);

       } # if class result

    } # while sample_flg

   } # for sample

   &::Logging("Learn Finish") if $debug == 1;

}



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
use DateTime;

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

       $self->{stat} = ""; # モジュールのステータス
                           # layer_inited : 初期化済  ->learn()　が動作する
                           # learned :　学習済   ->calc_multi()が動作する
			   #
       $self->{input} = undef ;
       $self->{learn_limit} = 10;   # 学習データ1個に対して、waitsの更新を制限する。しかし、waitsに変化がないと抜けるのでlimitになっていない
       $self->{learn_finish} = {};  #学習が終わるためのチェックリスト　ハッシュでclassラベルをチェックする

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
        $self->{initdata} = clone($_[0]);    
	$self->{layer_member} = $self->{initdata}->{layer_member};
	weaken($self->{layer_member});
	my @tmp = @{$self->{layer_member}};
	$self->{layer_count} = $#tmp;
	$self->{input_count} = $self->{initdata}->{input_count};
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
	    my $waits = $nodes[$n]->waits($self->{takeover}->{waitsdump}->{$l}->{$n});
	    my $bias = $nodes[$n]->bias($self->{takeover}->{biasdump}->{$l}->{$n});
	}
    }

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
	    }
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
	}

    }

    } # else
}

sub dump_structure {
    my $self = shift;
    # file dump data
    # make "dump_structure.txt" HASH ref

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
	}
    }
 
    my $dt = DateTime->now();

    my $dumpdata = { 
                    DateTime => $dt,
                    layer_init => $self->{initdata} ,
		    waitsdump => $waitsdump,
		    biasdump => $biasdump,
                    learn_rate => $learn_rate_dump,
                   };
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
            #my @learn_input = @{$_[0]};
            #$self->{learn_input} = \@learn_input;
            $self->{learn_input} = $_[0];  # learn内のみ変数
	} else {
            croak "input data format not match!";
	}
    } 

    my $debug = 0; # 0: off 1: on
    my $hand = 0; # 0: off 1: on  手動実行時にほしい表示 収束するか傾向を見る場合

    # チェック比較時に利用する配列
    my $fillARRAY = [];
    for my $l ( 0 .. $self->{layer_count} ){
        for my $n ( 0 .. $self->{layer_member}->[$l] ) {
            push(@{$fillARRAY->[$l]} , 1); 
        }
    }

    my @layer = @{$self->{layer}};   # layer内はPerceptronなので、メソッドと変数をを間違えないように

    my $loop = 0;
    # 入力して各層を計算していく
    for my $sample (@{$self->{learn_input}}) {
        $loop++;

        my $sample_flg = 1;
	my $sample_count = 0;
        while ( $sample_flg ) {   # simpleの時と違ってループは不要だった？ 偏微分は数回で収束してしまう

            if ($sample_count >= $self->{learn_limit} ) {
                &::Logging("learn limit over!");
		$sample_flg = 0;
		exit;
	    }
            $sample_count++;

            &::Logging("Loop: $loop start ------------------") if $debug == 1;

	    #my $out = []; #3次元に成るように各ノードの出力結果  $out->[レイヤー]->[ノード]

            $self->input($sample->{input});
	    my $out = $self->calc_multi('learn');

	    # 出力層の結果をsampleのclassラベルと比較する
	    my $outstring = join ("" , @{$out->[$self->{layer_count}]});
	    my $sampleclassstring = join ("" , @{$sample->{class}});

	    &::Logging("DEBUG: outstring: $outstring") if $debug == 1;
	    &::Logging("DEBUG: sampleclassstring: $sampleclassstring") if $debug == 1;

	    $self->{learn_finish}->{$sampleclassstring} ||= 0; # 無ければ0をセット  何度も上書きされるので注意
	    &::Logging("DEBUG: learn_finish: $sampleclassstring $self->{learn_finish}->{$sampleclassstring}") if $debug == 1 ;

	    if ($sampleclassstring eq $outstring) {
               # 出力結果が一致したら
	       # 一致いしないが、重みが更新されず変化なしでスルーすると何故か動作する。。。？？？

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
				if ( $out->[$l]->[$n] > 0 ) {
		                    $second = 1; # 活性化関数 の微分 ReLU関数
			        } else {
                                    $second = 0;
				}
				$third = $sample->{input}->[$w]; # 入力値 ########
				$backprobacation->[$l]->[$n]->[$w] = clone({ first => $first , second => $second }); # 次の回の計算で利用される
                                my $tmp = $old_layerwaits->[$l]->[$n]->[$w] - ( $learn_rate * $backprobacation->[$l+1]->[$self->{layer_member}->[$l+1]]->[$n]->{first} * $backprobacation->[$l+1]->[$self->{layer_member}->[$l+1]]->[$n]->{second} * $third );
                                push (@{$waits_delta} , $tmp); #調整したwaits

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
                                    $third = $out->[$l-1]->[$n];   #入力から得られた結果の偏微分 -> 前の層からの入力
				    $backprobacation->[$l]->[$n]->[$w] = clone({ first => $first , second => $second }); # 次の層の計算で利用される

				    &::Logging("DEBUG: learn_rate: $learn_rate first: $first second: $second third $third ") if $debug == 1;
                                    my $tmp = $old_layerwaits->[$l]->[$n]->[$w] - ( $learn_rate * $first * $second * $third ); 
                                    push (@{$waits_delta} , $tmp);  # 調整したwaits 

			        } elsif ( $l < $self->{layer_count} ) {
                                # 中間層
			        # waitsは後層すべてのノードに影響するので、合計が必要になる waitsの添字は現在のノード番号
			            for my $nsum ( 0 .. $self->{layer_member}->[$l+1] ) {
                                        $first += $backprobacation->[$l+1]->[$nsum]->[$n]->{first} * $backprobacation->[$l+1]->[$nsum]->[$n]->{second} * $new_layerwaits->[$l+1]->[$nsum]->[$n]; 
				    }
				    # ReLU関数の微分
				    if ( $out->[$l]->[$n] > 0 ) {
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
                                       $theta += $backprobacation->[$l+1]->[$node]->[$n]->{first};
				       $iota += $backprobacation->[$l+1]->[$node]->[$n]->{second};
                                    }
                                    my $tmp = $old_layerwaits->[$l]->[$n]->[$w] - ( $learn_rate * $theta * $iota * $third );
                                    push (@{$waits_delta} , $tmp); #調整したwaits

			       }
                            } # for $w  ############################################
		        } # if $l==0 esle

			# biasの更新 (ここはノード単位で1回)
			if ($l == $self->{layer_count} ) {
			    #　出力層
			    my $bias = $self->{layer}->[$l]->[$n]->bias();
			    my $tmp = $bias - ( $learn_rate *  ($out->[$l]->[$n] - $sample->{class}->[$n]) * $out->[$l]->[$n] * $out->[$l-1]->[$n]); 
			    $self->{layer}->[$l]->[$n]->bias($tmp);

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

            $self->{input} = clone($_[0]);

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

    my $out = []; #3次元に成るように各ノードの出力結果  $out->[レイヤー]->[ノード]
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
                } elsif ( $l < $self->{layer_count}) {
                #　中間レイアー
                    # 前段の出力を集計 
                    for my $node ( 0 .. $self->{layer_member}->[$l-1]) {   # $nだとややこしいのであえて書き方を変える
                        push(@{$l_input} , $out->[$l-1]->[$node]);
		    }
		    $layer[$l]->[$n]->input($l_input);
		    my $res = $layer[$l]->[$n]->calcReLU();
                    push(@{$out->[$l]} , $res);
	        } elsif ( $l == $self->{layer_count}) {
	        # 出力レイアー
                    for my $node ( 0 .. $self->{layer_member}->[$l-1]) {
                        push(@{$l_input} , $out->[$l-1]->[$node]);
		    }
	    #my $waits = $layer[$l]->[$n]->waits();
	    #say "l: $l n: $n l_input: @{$l_input} waits: @{$waits}";
                    $layer[$l]->[$n]->input($l_input);
                    my $res = $layer[$l]->[$n]->calcStep();
                    push(@{$out->[$l]} , $res);
	        }
	  } # for $n
    } # for $l

    # ARRAY ref
    return $out;
}

