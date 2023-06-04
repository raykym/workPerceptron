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

use Time::HiRes '/ time /';

use FindBin;
use lib "$FindBin::Bin/../lib";
use Sincpdl;
use TwoLayerNet;
    # 直接編集して、書き換えが必要！！！！！！！
    # 用途に応じて、Sigmoid_layer , IdentityWithLoss_layerの指定が必要
    # L2normの指定はパーッケージ側
use Adam_optimizer;

use Data::Dumper;

use Storable qw/ freeze thaw store retrieve/;
use PDL::IO::Storable;

# データ取得
my $trainMake = Sincpdl->new;
my ( $all_x , $all_t ) = $trainMake->make;

my $input_size = 2;
my $hidden_size = 500;
my $output_size = 1;
my $waits_init = "xavier";
my $L2norm = 0.0;
my $network = TwoLayerNet->new($input_size , $hidden_size , $output_size , $waits_init , $L2norm );
    # input_size , hidden_size , output_size , waits_init , weight_decay_rambda
    # 活性化関数はTwoLayerNetで直接指定
my @dims = $all_x->dims;
my $all_data_size = $dims[0]; # Sincpdlの最初の次元が個数になっているので
my $pickup_size = 24000; #データのピックアップ数
my $test_size = 100; #テストデータのピックアップ数
my $batch_size = 3000; #バッチ数
my $itre = $pickup_size / $batch_size ; # イテレーター数
my $epoch = 2; #エポック数
# L2normはTwoLayerNet.pmで決め打ちなので、そちらを編集する必要がある。

my $learn_rate = 0.001; # optimizerで指定する
my $optimizer = Adam_optimizer->new($learn_rate);

my $serialize = undef;

# サブ ルーチンエリア
sub Logging {
        my $logline = shift;
        my $dt = time();
        say "$dt | $logline";

        undef $dt;
        undef $logline;

        return;
}

# Sincpdlに組み込んだほうが良かったかな？
sub makeindex {
    my $pick_size = shift;
    # train_xからランダムにピックアップする（非復元抽出）為のindex作成 
    # epoch単位、pickup_sizeでデータは一意に成る。　epochを繰り返すと重複は起こり得る
    my @index = ();
    my @array = ();

    for my $i ( 0 .. $all_data_size -1) {  # この変数の使い方は良くないか。。。
        push(@array , [ $i , int(rand(100000)) ] ); # 乱数は$all_data_sizeよりも大きい数値
    } 
    my @array_sort = sort { $a->[1] <=> $b->[1] } @array; # 乱数側をソートすると、indexがシャッフルされる

    for my $idx ( 0 .. $pick_size - 1 ) {
        push ( @index , $array_sort[$idx]->[0]);
    }
    return @index;
}



# 学習の繰り返し回数 
for my $epoch_cnt ( 1 .. $epoch ) {

    my $x_batch = null;
    my $t_batch = null;
=pod
    my $pickup_idx = random($pickup_size); #ピックアップサイズのndarray (復元型抽出）
    my $pickup_idx = random(@index); 
       $pickup_idx = $pickup_idx * ($all_data_size - 1); # 最大値は全データ個数
       $pickup_idx = convert($pickup_idx,long); # 整数に
=cut
    my @index = &makeindex($pickup_size); # 非復元型抽出 エポック単位ではサンプルの重複はしない
    my $pickup_idx = pdl(@index);
    undef @index;

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
	#
	#say "itre: $idx";

        # 傾き計算
	my $grad = $network->gradient($x_batch , $t_batch);
	#my $grad = $network->numerical_gradient($x_batch , $t_batch);

        # 更新 
        $optimizer->update($network->{params} , $grad );

	   #  say Dumper $network->{params};

        my $loss = $network->loss($x_batch , $t_batch);
	#   $loss /= $batch_size; # なんとなく加えた そういうふうに読めたので
   #  say "itre: $idx loss: $loss ";

    } # for $idx
    
    # testデータの選択 (復元性抽出）
    my $test_idx = random($test_size); #ピックアップサイズのndarray
       $test_idx = $test_idx * ($all_data_size - 1); # 最大値は全データ個数
       $test_idx = convert($test_idx,long); # 整数に

    my $test_X_PDL = $all_x->index1d($test_idx)->sever;
    my $test_T_PDL = $all_t->index1d($test_idx)->sever;

    my $train_acc = $network->accuracy($pickup_X_PDL_T , $pickup_T_PDL_T);
    my $test_acc = $network->accuracy($test_X_PDL->transpose , $test_T_PDL->transpose);
    Logging("epoch: $epoch_cnt");
    Logging("$train_acc | $test_acc");

    #ガーベッジコレクション
    $serialize = freeze $network;
    $network = thaw($serialize);

} # for epoch


# 学習後に元データを入力して再現性を確認する
   open ( my $fh , '>' , "./sinc_plotdata.txt_U${hidden_size}_b${batch_size}_ep${epoch}_L2norm${L2norm}_pic${pickup_size}_lr${learn_rate}");
    # x,yを与えて結果をまとめてファイル出力,gnuplotで利用する

    for ( my $x = -10 ; $x <= 10 ; $x++  ) {
        for ( my $y = -10 ; $y <= 10 ; $y++  ) {
            my $RET = $network->predict(pdl([ $x , $y ])); 
	    #say "(loss)RET: $RET";
	    my @out = list($RET);
            say $fh " $x $y $out[0] ";
        }
    }

  close $fh;

  # イメージ取得
  my $hparams = {};
     $hparams->{input_size} = $input_size;
     $hparams->{hidden_size} = $hidden_size;
     $hparams->{output_size} = $output_size;
     $hparams->{waits_init} = $waits_init;
     $hparams->{L2norm} = $L2norm;
     $hparams->{network} = $network;
     $hparams->{all_data_size} = $all_data_size; 
     $hparams->{pickup_size} = $pickup_size; 
     $hparams->{test_size} = $test_size; 
     $hparams->{batch_size} = $batch_size;
     $hparams->{itre} = $itre; 
     $hparams->{epoch} = $epoch;
     $hparams->{learan_rate} = $learn_rate;
     $hparams->{optimizer} = $optimizer;

     store $hparams , "twolayernet.hparams_U${hidden_size}_b${batch_size}_ep${epoch}_L2norm${L2norm}_pic${pickup_size}_lr${learn_rate}";

