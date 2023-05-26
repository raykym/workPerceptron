#!/usr/bin/env perl
#
# twolayernetを使って数値微分とバックプロパゲーションを比較する

use strict;
use warnings;
use utf8;
use feature 'say';

binmode 'STDOUT' , ':utf8';

$|=1;

use PDL;
use PDL::Core ':Internal';
use PDL::NiceSlice;

use FindBin;
use lib "$FindBin::Bin/lib";
use MnistLoad;
use TwoLayerNet;

# MNISTファイルのロード
my ($train_x , $train_l , $test_x , $test_l ) = MnistLoad::mnistload();

# 標準化
$train_x = MnistLoad::normalize($train_x);
$test_x = MnistLoad::normalize($test_x);

#ラベルをhot-one化
$train_l = MnistLoad::chg_hotone($train_l);
$test_l = MnistLoad::chg_hotone($test_l);

my $network = TwoLayerNet->new(784 , 50 , 10);
#my $network = TwoLayerNet->new(784 , 50 , 1 , 'xavier');
              # input_size = 784 , hidden_size = 50 , output_size = 10

# データセットから4個をバッチとして分ける 分離する意味が在るかわからないけれど
my $x_batch = $train_x(:3)->sever;
my $t_batch = $train_l(:3)->sever;   # l -> t ラベルはターゲット

say "DEBUG: x_batch";
say $x_batch->shape; # ( 4 , 784)
say "DEBUG: t_batch";
say $t_batch->shape; # ( 4 , 784)
say "";
# なので転置する
$x_batch = $x_batch->transpose;
$t_batch = $t_batch->transpose;
say $x_batch->shape; # ( 784 , 4)
say $t_batch->shape; # ( 784 , 4)
say "";

# waitsのshapeもチェックしたほうが良いかな？
#

my $grad_numerical = $network->numerical_gradient($x_batch , $t_batch);

my $grad_backprop = $network->gradient($x_batch , $t_batch);

# 各重みの絶対誤差の平均を求める
#  $grad_...はハッシュにPDLが入っている

for my $key ( keys %{$grad_numerical} ) {
=pod
    say "key: $key";
    say "backprop";
    say $grad_backprop->{$key}->shape;
    say "numerical";
    say $grad_numerical->{$key}->shape;
=cut
   my $TMP = abs ( $grad_backprop->{$key} - $grad_numerical->{$key} );
   # my $TMP = $grad_backprop->{$key} - $grad_numerical->{$key};
    my $diff = $TMP->avg;
    say "$key : $diff";
    say "";
}
# 比較誤差は本に書いてあるほどに精度が小さく無い？？？計算が間違っている？





