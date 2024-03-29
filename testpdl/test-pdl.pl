#!/usr/bin/env perl
#
# 多層パーセプトロンのPDLを利用した版
# ゼロから作るDeepLearningを参考にしながら
# perl PDLに置き換えていく

use strict;
use warnings;
use utf8;
#use feature 'say';
use v5.32;

binmode 'STDOUT' , ':utf8';

use FindBin;
use lib "$FindBin::Bin/../lib";
use Multilayer_PDL;
#use TwoLayerNet;

use Tie::IxHash;
use Time::HiRes qw / time /;

use PDL;
use PDL::NiceSlice;
use PDL::Core ':Internal';

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

sub function_2 {
    my $X = shift;
    $X = topdl($X);

    return $X(0) ** 2 + $X(1) ** 2;
}


=pod

my $Multilayer = Multilayer_PDL->new;

my $X = pdl ( -3.0 , 4.0 );
my $lr = 0.1;
my $step_num = 100;


#Logging &function_2($X);
#Logging $Multilayer->numerical_gradient(\&function_2 , $X );

Logging $Multilayer->gradient_descent(\&function_2 , $X , $lr , $step_num);

=cut

#=pod
Logging("TwoLayerNet");

my $net = TwoLayerNet->new(784 , 100 , 10 );
    # input_size = 784 , hidden_size = 100, output_size =10

say $net->{params}->{W1}->shape; # ( 100 , 784)
say $net->{params}->{b1}->shape; # (100)
say $net->{params}->{W2}->shape; # (10,100)
say $net->{params}->{b2}->shape; # (10)



Logging("predict");

my $X = random(100,784); #　ダミー入力データ　100枚
   $X = $X->transpose; # 入力データは転置させる。

my $Y = $net->predict($X);

#say $Y;


Logging("gradient");

   $X = random(100,784) ; # ダミー入力データ
   $X = $X->transpose;
my $T = random(100,10); # ダミー正解データ
   $T = $T->transpose;

my $grads = $net->numerical_gradient($X,$T);

say $grads->{W1}->shape; # (100,784)
say $grads->{b1}->shape; # (100)
say $grads->{W2}->shape; # (10,100)
say $grads->{b2}->shape; # (10)

#=cut

=pod
Logging("Buy apple");
Logging("(forward)");
my $apple = 100;
my $apple_num = 2;
my $tax = 1.1;


my $mul_apple_layer = MulLayer->new;
my $mul_tax_layer = MulLayer->new;

my $apple_price = $mul_apple_layer->forward($apple, $apple_num);
my $price = $mul_tax_layer->forward($apple_price , $tax);

say $price; # 220;

Logging("(backward)");

my $dprice = 1;
my ($dapple_price , $dtax ) = $mul_tax_layer->backward($dprice);
my ($dapple , $dapple_num ) = $mul_apple_layer->backward($dapple_price);

say "$dapple | $dapple_num | $dtax "; # 2.2 110 200
=cut

=pod
Logging("Buy apple orange");

my $apple = 100;
my $apple_num = 2;
my $orange = 150;
my $orange_num = 3;
my $tax = 1.1;

my $mul_apple_layer = MulLayer->new;
my $mul_orange_layer = MulLayer->new;
my $add_apple_orange_layer = AddLayer->new;
my $mul_tax_layer = MulLayer->new;

Logging("(forward)");
my $apple_price = $mul_apple_layer->forward($apple , $apple_num);
my $orange_price = $mul_orange_layer->forward($orange , $orange_num);
my $all_price = $add_apple_orange_layer->forward($apple_price, $orange_price);
my $price = $mul_tax_layer->forward($all_price , $tax);

Logging("(backword)");
my $dprice = 1;
my ($dall_price , $dtax ) = $mul_tax_layer->backward($dprice);
my ($dapple_price , $dorange_price) = $add_apple_orange_layer->backward($dall_price);
my ($dorange , $dorange_num) = $mul_orange_layer->backward($dorange_price);
my ($dapple , $dapple_num) = $mul_apple_layer->backward($dapple_price);

say $price;
say "$dapple_num | $dapple | $dorange | $dorange_num | $dtax ";

=cut










package TwoLayerNet;

# numPyとPDLの仕様の違い (行、列）と（列、行）については 入力前に転置する、waitsは転置しておく

use utf8;
binmode 'STDOUT' , ':utf8';

use PDL;
use PDL::Core ':Internal';
use PDL::NiceSlice;

use FindBin;
use lib "$FindBin::Bin/../lib";
use Multilayer_PDL;
use Ml_functions;

use Tie::IxHash;
use Time::HiRes qw / time /;



sub new {
    my $proto = shift;
    my $class = ref $proto || $proto;
    my ( $input_size , $hidden_size , $output_size , $weight_init_std ) = @_;
       $weight_init_std = 0.01 if ! defined $weight_init_std;

    my $rng = PDL::GSL::RNG->new('mt19937_1999');
       $rng->set_seed(time());

    my $self = {};
       $self->{params}->{W1} = $weight_init_std * $rng->ran_gaussian(1 ,$input_size , $hidden_size );
       $self->{params}->{W1} = $self->{params}->{W1}->transpose; # waitsは転置する
       $self->{params}->{b1} = zeros($hidden_size);
       $self->{params}->{W2} = $weight_init_std * $rng->ran_gaussian(1 ,$hidden_size , $output_size );
       $self->{params}->{W2} = $self->{params}->{W2}->transpose; 
       $self->{params}->{b2} = zeros($output_size);

     bless $self , $class;

     return $self;
}

sub predict {
    my $self = shift;
    my $X = shift;
    $X = topdl($X);
=pod
    say "DEBUG: predict: X";
    say $X->shape;
    say "";
=cut
    my ($W1 , $W2) = ( $self->{params}->{W1} , $self->{params}->{W2} );
    my ($b1 , $b2) = ( $self->{params}->{b1} , $self->{params}->{b2} );

    my $A1 = $X x $W1 + $b1;
    my $Z1 = Ml_functions::sigmoid($A1); 
    my $A2 = $Z1 x $W2 + $b2;
    my $y = Ml_functions::softmax($A2); 

       return $y;
}

sub loss {
    my ($self , $X , $T ) = @_;
=pod
    say "DEBUG: loss: X";
    say $X->shape;
    say "DEBUG: loss: T";
    say $T->shape;
    say "";
=cut

    my $Y = $self->predict($X);

    return Ml_functions::cross_entropy_error($Y , $T);
}

sub accuracy {
    my ( $self , $X , $T ) = @_;

    my $Y = $self->predict($X);
    $Y .= argmax($Y);
    $T .= argmax($T);

    # 入れ替えのタイミングでこんがらがるな。。。どうすれば
    #my $accuracy = sum($Y == $T ) / float($X->shape(0)); # 直訳
    my @shape = $X->dims; #(列、行) 行を指定する
    my $accuracy = sum($Y == $T ) / float($shape[1]); 

    undef @shape;
    undef $Y;
    undef $T;

    return $accuracy;
}

sub numerical_gradient {
    my ( $self , $X , $T ) = @_;

    say "DEBUG: numerical_gradient: x";
    say $X->shape;
    say "DEBUG: numerical_gradient: T";
    say $T->shape;
    say "";

    my $loss_W = sub {
	    # 動作がおかしいのでサブルーチンの引数を取らないことにする
	    #my ( $x , $T ) = @_;
	    #$x = topdl($x);

=pod
                     say "DEBUG: loss_W: X";
		     say $X->shape;
		     say "DEBUG: loss_W: T";
		     say $T->shape;
		     say "";
=cut
		     # ここも関数のカプセル化を無視すればなくても良いはず
		     #$x = $x->transpose; #何故が転置が解除されるので
		     #state $X = $x; # 初期値で固定する
                     
		     $self->loss($X,$T);

                     };

    my $grads = {};
       $grads->{W1} = Ml_functions::numerical_gradient($loss_W , $self->{params}->{W1} );
       # 2回めのときに、$loss_Wの引数に{b1}が入ってしまうのでstateで固定
       $grads->{b1} = Ml_functions::numerical_gradient($loss_W , $self->{params}->{b1} );
       $grads->{W2} = Ml_functions::numerical_gradient($loss_W , $self->{params}->{W2} );
       $grads->{b2} = Ml_functions::numerical_gradient($loss_W , $self->{params}->{b2} );

    undef $loss_W; # 呼び出し毎にstateをリセットする

    return $grads;

}






package MulLayer;

use utf8;
binmode 'STDOUT' , ':utf8';

use PDL;
use PDL::Core ':Internal';
use PDL::NiceSlice;

use FindBin;
use lib "$FindBin::Bin/../lib";
use Multilayer_PDL;
use Ml_functions;

use Tie::IxHash;
use Time::HiRes qw / time /;




sub new {
    my $proto = shift;
    my $class = ref $proto || $proto;

    my $self = {};
       $self->{X} = undef;
       $self->{Y} = undef;

    bless $self , $class;


    return $self;

}

sub forward {
    my ($self , $X , $Y) = @_;

    $self->{X} = $X;
    $self->{Y} = $Y;

    my $OUT = $X * $Y;

    return $OUT;

}

sub backward {
    my ($self , $dout ) = @_;

    my $DX = $dout * $self->{Y}; # X,Yをひっくり返す
    my $DY = $dout * $self->{X};

    return ( $DX , $DY );

}



package AddLayer;

use utf8;
binmode 'STDOUT' , ':utf8';

use PDL;
use PDL::Core ':Internal';
use PDL::NiceSlice;

use FindBin;
use lib "$FindBin::Bin/../lib";
use Multilayer_PDL;
use Ml_functions;

use Tie::IxHash;
use Time::HiRes qw / time /;

sub new {
    my $proto = shift;
    my $class = ref $proto || $proto;

    my $self = {};

    bless $self , $class;

    return $self;
}

sub forward {
    my ( $self , $X , $Y ) = @_;

    my $OUT = $X + $Y;

    return $OUT;
}

sub backward {
    my ( $self , $dout ) = @_;

    my $DX = $dout * 1;
    my $DY = $dout * 1;

    return ($DX , $DY);
}


