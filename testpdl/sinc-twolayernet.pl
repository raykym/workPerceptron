#!/usr/bin/env perl
#
# sincを学習して再現性を試す。　普遍性の実験
#

use v5.32;
use utf8;

binmode 'STDOUT' , ':utf8';

use PDL;
use PDL::Core ':Internal';
use PDL::NiceSlice;

use FindBin;
use lib "$FindBin::Bin/../lib";
use Sincpdl;
use TwoLayerNet;
    # 直接編集して、書き換えが必要！！！！！！！
    # 用途に応じて、Sigmoid_layer , IdentityWithLoss_layerの指定が必要
use Adam_optimizer;

use Data::Dumper;

# データ取得
my $trainMake = Sincpdl->new;
my ( $all_x , $all_t ) = $trainMake->make;

my $network = TwoLayerNet->new(2 , 500 , 1 , 'xavier');

my @dims = $all_x->dims;
my $all_data_size = $dims[0]; # Sincpdlの最初の次元が個数になっているので
my $pickup_size = 10000; #データのピックアップ数
my $test_size = 100; #テストデータのピックアップ数
my $batch_size = 100; #バッチ数
my $itre = $pickup_size / $batch_size ; # イテレーター数
my $epoch = 300; #エポック数

# 学習の繰り返し回数 
for my $epoch_cnt ( 1 .. $epoch ) {

    my $x_batch = null;
    my $t_batch = null;

    my $pickup_idx = random($pickup_size); #ピックアップサイズのndarray
       $pickup_idx = $pickup_idx * ($all_data_size - 1); # 最大値は全データ個数
       $pickup_idx = convert($pickup_idx,long); # 整数に

    my $pickup_X_PDL = $all_x->index1d($pickup_idx)->sever;
    my $pickup_T_PDL = $all_t->index1d($pickup_idx)->sever;
    
    # accuraryで利用する転置
    my $pickup_X_PDL_T = $pickup_X_PDL->copy;
       $pickup_X_PDL_T = $pickup_X_PDL->transpose;
    my $pickup_T_PDL_T = $pickup_T_PDL->copy;
       $pickup_T_PDL_T = $pickup_T_PDL->transpose;

    for (my $idx=0 ; $idx <= $itre -1 ; $idx++ ) {
        # バッチに切り出し
        $x_batch = $pickup_X_PDL->range($idx * $batch_size , $batch_size)->sever;
        $t_batch = $pickup_T_PDL->range($idx * $batch_size , $batch_size)->sever;

        $x_batch = $x_batch->transpose;
        #t_batchは1次元なのでパス

        # 傾き計算
	my $grad = $network->gradient($x_batch , $t_batch);
	#my $grad = $network->numerical_gradient($x_batch , $t_batch);

        # 更新 
        my $optimizer = Adam_optimizer->new();
           $optimizer->update($network->{params} , $grad );

	   #  say Dumper $network->{params};

        my $loss = $network->loss($x_batch , $t_batch);
	   $loss /= $batch_size; # なんとなく加えた そういうふうに読めたので
   #  say "itre: $idx loss: $loss ";

    } # for $idx
    
    # testデータの選択
    my $test_idx = random($test_size); #ピックアップサイズのndarray
       $test_idx = $test_idx * ($all_data_size - 1); # 最大値は全データ個数
       $test_idx = convert($test_idx,long); # 整数に

    my $test_X_PDL = $all_x->index1d($test_idx)->sever;
    my $test_T_PDL = $all_t->index1d($test_idx)->sever;

    my $train_acc = $network->accuracy($pickup_X_PDL_T , $pickup_T_PDL_T);
    my $test_acc = $network->accuracy($test_X_PDL->transpose , $test_T_PDL->transpose);
    say "epoch: $epoch_cnt";
    say "$train_acc | $test_acc";

} # for epoch


# 学習後に元データを入力して再現性を確認する
   open ( my $fh , '>' , './sinc_plotdata.txt');
    # x,yを与えて結果をまとめて出力をgnuplotでプロットさせる

    for ( my $x = -10 ; $x <= 10 ; $x++  ) {
        for ( my $y = -10 ; $y <= 10 ; $y++  ) {
            my $RET = $network->predict(pdl([ $x , $y ])); 
	    #say "(loss)RET: $RET";
	    my @out = list($RET);
            say $fh " $x $y $out[0] ";
        }
    }

    close $fh;


