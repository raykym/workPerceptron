package Multilayer_PDL;

#
# PDLの行列演算を利用したマルチパーセプトロン
# ゼロから作るDeep Leaningを元に作成

use Carp;
use PDL;
use PDL::IO::GD;
use PDL::GSL::RNG;
use PDL::NiceSlice;
use PDL::Core ':Internal';

use FindBin;;
use lib "$FindBin::Bin";
use Datalog;




sub new {
    my $proto = shift;
    my $class = ref $proto || $proto;

    my $self = {};

    bless $self , $class;
    return $self;

}

# 大文字変数はndarrayを想定
# 元が関数型なのでOOの書式でselfを追加している
sub numerical_gradient {
    my ($self , $func , $X ) = @_;
    # $funcはサブルーチンのリファレンス
    $X = topdl($X);

    my $h = 0.00001;
    my $GRAD = zeros($X);

    my $cnt = $X->nelem;
       $cnt--;

    for my $idx ( 0 .. $cnt) {
        my $tmp_val = $X->index($idx)->sever;   # ndarrayからデータを抜き出す場合はseverで切断しないとリファレンスになってしまう。
	#my $tmp_val = $X($idx);

	$X($idx) .= $tmp_val + $h;
	$fxh1 = &{$func}($X);
	#&::Logging("DEBUG: idx: $idx fxh1: $fxh1");

	$X($idx) .= $tmp_val - $h;
	my $fxh2 = &{$func}($X);
	#&::Logging("DEBUG: idx: $idx fxh2: $fxh2");

	$GRAD($idx) .= ($fxh1 - $fxh2) / (2 * $h);
	$X($idx) .= $tmp_val;
    }
    undef $tmp_val;
    undef $cnt;
    undef $X;
    undef $fxh1;
    undef $fxh2;

    return $GRAD;
}

# モジュールのメソッドでは関数の引数に与えると上手く動かない
sub function_2 {
    my ($self , $X ) = @_;
    $X = topdl($X);
    # my $ret = $X(0) ** 2 + $X(1) ** 2;
    return $X(0) ** 2 + $X(2) ** 2 ;
}

sub gradient_descent{
    my ( $self , $func , $X , $lr , $step_num ) = @_;

    $X = topdl($X);

    for my $cnt ( 0 .. $step_num-1) {
        my $GRAD = $self->numerical_gradient($func , $X);
	$X -= $lr * $GRAD;
    }
    return $X;

}



1;
