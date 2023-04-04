package Perceptron;
# シンプルなパーセプトロンのモデルを作ってみる

use Carp;
use FindBin;
use lib "$FindBin::Bin";
use SPVM 'Util';
use Clone qw/clone/;
use Scalar::Util qw/ weaken /;
use feature 'say';

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
       $self->{tmp} = undef;
       $self->{calc_sum} = undef;

       bless $self , $class;

    return $self;
}

sub bias {
    my $self = shift;
    # 乱数を自動入力される想定

    if (@_) {
        if ( ref $_[0] ) {
            croak "input is reference";
	}
           $self->{bias} = $_[0];
    }

    return $self->{bias};
}

sub waits {
    my $self = shift;
    # waitの入力を一括で行う場合
    # biasを含まない想定
    #
    my @waits = (); # 保持し続ける為

    if (@_) {
        if ( $_[0] =~ /ARRAY/ ) {
            $self->{waits} = $_[0];
	} elsif ( $_[0] =~ /HASH/) {
	    croak "miss input HASH!";
	} else {
            @waits = @_;
	    $self->{waits} = \@waits;
	}
    } # if @_
    #  04_memoryleark.tを実行しても終わらない
    #  undef @waits; # これを入れると無限ループしてメモリーリークが見える

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
    # perceptron単体用

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

    undef @waits;
    undef @input;

    if ( $sum >= $self->{bias} ){
        return 1;
    } elsif ( $sum < $self->{bias} ) {
        return -1;
    }
}

# 活性化関数を分離した書き方
sub calcSum {
    my $self = shift;
 
    my @waits = @{$self->{waits}};
    my @input = @{$self->{input}};

    if ($#waits != $#input ) {
        croak "wait input miss match!  $#waits | $#input ";
	exit;
    }

    my $sum = 0;

    $sum = SPVM::Util->onedinnersum($self->{input} , $self->{waits});

    undef @waits;
    undef @input;

    $self->{calc_sum} = $sum;  # biasを除く

    return $self; # ->calcsum->ReLU() の為
}

sub ReLU {
    my $self = shift;

    my $tmp = $self->{calc_sum}; 
       $tmp += $self->{bias};

    if ( $tmp > 0 ){
        return $tmp;
    } elsif ( $tmp <= 0 ) {
        return 0;
    }

}

# ReLUのデバッグ用 -> 活性化関数無し
sub None {
    my $self = shift;
    # biasの判定をスルー

    my $tmp = $self->{calc_sum} + $self->{bias};
    # 上の式の記号を＋にするとwaitsの計算と同じなのだけど、学習率を何に設定しても終わらなくなる
    # しきい値としてbiasをにんしきしている場合はマイナスだが、入力に比例するように見える
    # 活性化関数を加味しないとすると、、、、

    return $tmp;
}

sub Step {
    my $self = shift;

    if ( $self->{calc_sum} >= $self->{bias} ){
        return 1;
    } elsif ( $self->{calc_sum} < $self->{bias} ) {
        return 0;
    }
}

sub Sigmoid {
    my $self = shift;

    my $tmp = $self->{calc_sum}; 
       $tmp += $self->{bias};

    my $sig = 1 / ( 1 + exp(-$tmp) );
    undef $tmp;
    return $sig;
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

    undef @waits;
    undef @input;

    $self->{calc_sum} = $sum; 

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

    undef @waits;
    undef @input;

    $self->{calc_sum} = $sum; 

    # step関数
    if ( $sum >= $self->{bias} ){
        return 1;
    } elsif ( $sum < $self->{bias} ) {
        return 0;
    }
}

sub calcSigmoid {
    my $self = shift;
    # 未使用

    my @waits = @{$self->{waits}};
    my @input = @{$self->{input}};

    if ($#waits != $#input ) {
        croak "wait input miss match!  $#waits | $#input";
	exit;
    }

    my $sum = 0;

    $sum = SPVM::Util->onedinnersum($self->{input} , $self->{waits});

    undef @waits;
    undef @input;

    $self->{calc_sum} = $sum; 

    # sigmoid関数
    my $sig = 1 / ( 1 + exp(-$sum) );

    if ( $sig >= $self->{bias} ){
        return 1;
    } elsif ( $sig < $self->{bias} ) {
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
        if ( $_[0] =~ /ARRAY/ || $_[0] =~ /HASH/ ) {
            croak "input not number";
	}

        if ($_[0] =~ /\d?/) {
            my $cnt = $_[0]; # waitの数
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


	    if ( defined $node_count ) {
                my $rand = rand( 2 / $node_count );
                $self->bias($rand);
	    } else {
                my $rand = rand(1);
                $self->bias($rand);
            }

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

       $self->{tmp} = [];
       @{$self->{tmp}} = @{$sample->{input}}; # $sampleはforのローカル変数なのでデリファレンスして実体化させる
       # forループのローカル変数にリファレンスを取って、そのまま参照すると、メモリーリークする
       my $sum = SPVM::Util->onedinnersum($sample->{dummy} , $self->{waits});   
       undef $self->{tmp}; # 使い終わったら削除する

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

	  #my @waits = map {$_ += ($delta / $_) } @{$self->{waits}}; # 以下と同等
	  my $SPVM_waits = SPVM::Util->map_learnsimple($delta , $self->{waits});
	  my $tmp = $SPVM_waits->to_elems();
	  my @waits = @{$tmp};
	  undef $SPVM_waits;

	  $self->waits(@waits);

       } # if class result

    } # while sample_flg

   } # for sample

   undef @learn_input;

   &::Logging("Learn Finish") if $debug == 1;

}

sub calc_sum {
    my $self = shift;
    # calcStep, calcReLUの場合、sumの値を記録しておく

    return $self->{calc_sum};
}

sub dummy_method {
    my $self = shift;

    my $tmp = $_[0];

    for my $dum ( 1 .. 10000 ) {
        # dummy
    }

    #  undef $tmp;

    croak "Allways on error Logic";

}

1;
