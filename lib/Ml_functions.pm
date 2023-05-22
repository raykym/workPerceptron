package Ml_functions;

# ライブラリパッケージ(サブルーチン集）
#
# 基本行列出力の部分は、転置済で入力される事を前提とする。
# numPyは（行、列）だが、PDLは（列、行）なのでPDLを基準とする
# waitの添字の向きを注意する

use utf8;
use PDL;
use PDL::Core ':Internal';
use PDL::NiceSlice;

use feature 'say';;

use lib '/home/debian/perlwork/work/workPerceptron/lib';
use Pdlitre;

sub sigmoid{
    my $X = shift;
    $X = topdl($X);

    return 1 / ( 1 + exp(-$X));
}

sub sigmoid_grad {
    my $X = shift;
    $X = topdl($X);

    return (1.0 - &sigmoid($X) ) * &sigmoid($X);
}

sub relu {
    my $X = shift;
    $X = topdl($X);
    
    # argmaximum(0,X)の置き換え
    my $z = zeros($X);
    my $cmp = cmpvec($z , $X);
    undef $z;
    if ( $cmp == -1 ) {
	# $zが小さいので$Xを返す
        return $X;
    } elsif ( $cmp >= 0 ) {
	#$zが等しいか大きいのでzeroを返す
        return zeros($X);
    }
}

sub relu_grad {
    my $X = shift;
    $X = topdl($X);

    # $Xが0より大きい部分を1としてndarrayを返す
    my $GRAD = zeros($X);
    return $GRAD > $X;
}

sub softmax {
    my $X = shift;
    $X = topdl($X);

    $X -= max($X); # オーバーフロー対策　最大値を全体から引く

    return exp($X) / sum( exp($X) );

}

sub sum_squared_error {
    my ($Y , $T) = @_;
    $Y = topdl($Y);
    $T = topdl($T);

    return 0.5 * sum(($Y-$T)**2);
}

sub cross_entropy_error {
    #入力行列は(列、行)形式を想定 転置済で入力される事を前提
    my ( $Y , $T ) = @_;
    $Y = topdl($Y);
    $Y_tmp = $Y->copy;
    $T = topdl($T);
    $T_tmp = $T->copy;

    # PDLの場合この変換処理は不要なのでは？　カラムリストに変換しているから
    #$Y->reshape(1 , $Y->nelem); # 直訳の書式
    $Y_tmp->reshape($Y_tmp->nelem); # 実際にやりたいことはこちら PDLはカラム配列が基本
    #$T->reshape(1 , $T->nelem);
    $T_tmp->reshape($T_tmp->nelem);

    if ( $Y_tmp->nelem == $T_tmp->nelem ) {

        $T_tmp = &argmax($T_tmp);
=pod
	# np.argmaxの置き換え
        my $f = ones($T);
	   $f *= max($T); #最大値と同型
	   $T .= $f == $T; #比較して最大値のインデックスが1で表示されたndarray
	   undef $f;
=cut
    }

    my $batch_size = $Y_tmp->shape(); #sample: [ 100 ]
    return -sum(log($Y_tmp + 0.00000001) * $T_tmp) / $batch_size; 

}

sub softmax_loss {
    my ( $X , $T ) = @_;
    $X = topdl($X);
    $T = topdl($T);

    my $Y = softmax($X);
    return cross_entropy_error($Y,$T);
}

sub numerical_gradient {
    my ($func , $X ) = @_;
    $X = topdl($X);
=pod
    say "DEBUG: ML_numerical_gradient: X";
    say $X->shape;
    say "";
=cut

    my $h = 0.0001;

    my $grad = zeros($X);

    my $it = Pdlitre->new($X);

    while ($it->finished) {

        my @idx = $it->multi_index;
	my $TMP_VAL = $X(@idx)->sever;
	   $X(@idx) += $h;
        my $fxh1 = &{$func}($X);

	   $X(@idx) .= $TMP_VAL;
           $X(@idx) -= $h;
	my $fxh2 = &{$func}($X);

	   $grad(@idx) .= ($fxh1 - $fxh2) / (2*$h);

	   $X(@idx) .= $TMP_VAL;

	   $it->itrenext;
    }

    return $grad;

}




# 自作
# numPyのメソッドをここに作成
# 最大値のインデックスを返す
# pythonのaxis=1方向  列基準
sub argmax {
    my $X  = shift;
       $X = topdl($X);

    my $f = ones($X);
       $f *= max($X); #最大値と同型

    return $f == $X; #比較して最大値のインデックスが1で表示されたndarray
}


1;
