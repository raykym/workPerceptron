#!/usr/bin/env perl
#
# TwoLayerNetの実行版

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
use lib "$FindBin::Bin/../lib";
use MnistLoad;
use TwoLayerNet;

# MNISTファイルのロード
my ($train_x , $train_t , $test_x , $test_t ) = MnistLoad::mnistload();

# 標準化
$train_x = MnistLoad::normalize($train_x);
$test_x = MnistLoad::normalize($test_x);

#ラベルをhot-one化
$train_t = MnistLoad::chg_hotone($train_t);
$test_t = MnistLoad::chg_hotone($test_t);

# batchの選択に問題が起きるので転置済を用意する accuracyで利用する
my $train_x_T = $train_x->copy;
   $train_x_T = $train_x_T->transpose;
my $train_t_T = $train_t->copy;
   $train_t_T = $train_t_T->transpose;
my $test_x_T = $test_x->copy;
   $test_x_T = $test_x_T->transpose;
my $test_t_T = $test_t->copy;
   $test_t_T = $test_t_T->transpose;

my $network = TwoLayerNet->new(784 , 50 , 10 );
              # input_size = 784 , hidden_size = 50 , output_size = 10

my $itres_num = 10000;
my @dims = $train_x->dims; # (60000,784)
my $train_size = $dims[0]; # 転置前なので列が個数
my $batch_size = 100;
my $learning_rate = 0.1;

my $train_loss_list = []; # perl配列 
my $train_acc_list = [];
my $test_acc_list = [];
           #PDLの書式を勘違いするのか？ clumpコマンドが無いというメッセージが出る
#my $itre_per_epoch = max($train_size / $batch_size , 1); #　大きい方のどちらか
my $itre_per_epoch = ($train_size / $batch_size) < 1 ? 1 : ($train_size/$batch_size);
say "DEBUG: itre_per_epoch: $itre_per_epoch";

for my $i ( 0 .. $itres_num ) {
    # バッチ用抽出  np.random.choiceの代わり
    my $x_batch = null;
    my $t_batch = null;

    my $x_idx = random($batch_size); #バッチサイズのndarray
       $x_idx = $x_idx * ($train_size - 1); # 最大値はtrain_size
       #  say "x_idx shape";
       #  say $x_idx->shape;
       $x_idx = convert($x_idx,long); # 整数に均す

       $x_batch = $train_x->index1d($x_idx)->sever; # インデックスでバッチを切り出し
       $t_batch = $train_t->index1d($x_idx)->sever; 

       # 入力前に転置する
       $x_batch = $x_batch->transpose; 
       $t_batch = $t_batch->transpose;
       #  say "x_batch & t_batch shape";
       #  say $x_batch->shape;
       #  say $t_batch->shape;

    # 傾き計算
    my $grad = $network->gradient($x_batch , $t_batch);

    # 更新 これはアクセサーを用意していなかった。。。
    for my $key ( 'W1' , 'b1' , 'W2' , 'b2' ) {
	#say "key: $key";
	#say $grad->{$key}->shape; 
        $network->{params}->{$key} -= $learning_rate * $grad->{$key};
    }

    my $loss = $network->loss($x_batch , $t_batch);
    push(@{$train_loss_list} , $loss );

    if ( $i % $itre_per_epoch == 0 ) {
        say "itre_per_epoch i: $i";
        # データセットは転置済みを利用する PDLの仕様のため
        my $train_acc = $network->accuracy($train_x_T , $train_t_T);
	my $test_acc = $network->accuracy($test_x_T , $test_t_T);
	push(@{$train_acc_list}, $train_acc);
	push(@{$test_acc_list}, $test_acc);
        say "$train_acc | $test_acc";
    }


} # for $i

# なぜかsegment faualtで終わる
# とりあえず処理は一通り流れる


