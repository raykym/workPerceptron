#!/usr/bin/env perl
#
# twolayernetを使って数値微分とバックプロパゲーションを比較する

use v5.32;
use utf8;

binmode 'STDOUT' , ':utf8';

$|=1;

use PDL;
use PDL::Core ':Internal';
use PDL::NiceSlice;

use Storable qw/ freeze thaw store retrieve/;
use PDL::IO::Storable;

use FindBin;
use lib "$FindBin::Bin/../lib";
use MnistLoad;
use MultiLayerNet;
use Adam_optimizer;

# MNISTファイルのロード
my ($train_x , $train_t , $test_x , $test_t ) = MnistLoad::mnistload();

# 標準化
$train_x = MnistLoad::normalize($train_x);
$test_x = MnistLoad::normalize($test_x);

#ラベルをhot-one化
$train_t = MnistLoad::chg_hotone($train_t);
$test_t = MnistLoad::chg_hotone($test_t);

=pod
say "train x";
say $train_x->dims;
say "test x";
say $test_x->dims;
say "train t";
say $train_t->dims;
say "test t";
say $test_t->dims;

exit;
=cut

my @dims_x = $train_x->dims;
my @dims_t = $train_t->dims;

# パラメータ
my $input_size = $dims_x[1];
my $hidden_size =  [ 100 , 100 ];
my $output_size = $dims_t[1];
my $activation = 'relu';   # relu or sigmoid
my $waits_init = "he";     # xavier or he
my $L2norm = 0.0;
my $loss_func = 'cross_entropy_error';  # mean_squared_error or cross_entropy_error
my $network = MultiLayerNet->new($input_size , $hidden_size , $output_size , $activation , $waits_init , $L2norm , $loss_func);

my $all_data_size = $dims_x[0]; # Mnistloadの最初の次元が個数になっているので
my $pickup_size = 20000; #データのピックアップ数
my $test_size = 1000; #テストデータのピックアップ数
my $batch_size = 1000; #バッチ数
my $itre = $pickup_size / $batch_size ; # イテレーター数
my $epoch = 10; #エポック数

my $learn_rate = 0.001; # optimizerで指定する
my $optimizer = Adam_optimizer->new($learn_rate);

my $serialize = undef;

undef @dims_x;
undef @dims_t;


# サブ ルーチンエリア
sub Logging {
        my $logline = shift;
        my $dt = time();
        say "$dt | $logline";

        undef $dt;
        undef $logline;

        return;
}

sub makeindex {
    my $pickup_size = shift;
    # train_xからランダムにピックアップする（非復元抽出）為のindex作成 
    # epoch単位、pickup_sizeでデータは一意に成る。　epochを繰り返すと重複は起こり得る
    my @index = ();
    my @array = ();

    for my $i ( 0 .. $all_data_size -1) {  # この変数の使い方は良くないか。。。
        push(@array , [ $i , int(rand($all_data_size * 10)) ] ); # 乱数は$all_data_sizeよりも大きい数値
    }
    my @array_sort = sort { $a->[1] <=> $b->[1] } @array; # 乱数側をソートすると、indexがシャッフルされる

    for my $idx ( 0 .. $pickup_size - 1 ) {
        push ( @index , $array_sort[$idx]->[0]);
    }
    return @index;
}

# 学習の繰り返し回数 
for my $epoch_cnt ( 1 .. $epoch ) {
    Logging("epoch: $epoch_cnt");

    my $x_batch = null;
    my $t_batch = null;

    my @index = &makeindex($pickup_size); # 非復元型抽出 エポック単位ではサンプルの重複はしない
    my $pickup_idx = pdl(@index);
    undef @index;

    my $pickup_X_PDL = $train_x->index1d($pickup_idx)->sever;
    my $pickup_T_PDL = $train_t->index1d($pickup_idx)->sever;

    # accuraryで利用する転置
    my $pickup_X_PDL_T = $pickup_X_PDL->copy;
       $pickup_X_PDL_T = $pickup_X_PDL->transpose;
    my $pickup_T_PDL_T = $pickup_T_PDL->copy;
       $pickup_T_PDL_T = $pickup_T_PDL->transpose;

    for (my $idx=0 ; $idx <= $itre -1 ; $idx++ ) {
        # バッチに切り出し
        $x_batch = $pickup_X_PDL->range($idx * $batch_size , $batch_size)->sever;
        $t_batch = $pickup_T_PDL->range($idx * $batch_size , $batch_size)->sever;

        $x_batch = $x_batch->transpose; # shape (データ列 ,  データ個数)
        $t_batch = $t_batch->transpose;

        # 傾き計算
        my $grad = $network->gradient($x_batch , $t_batch);
        #my $grad = $network->numerical_gradient($x_batch , $t_batch);

        # 更新 
        $optimizer->update($network->{params} , $grad );

    } # for idx

   # testデータの選択 (復元性抽出）
    my $test_idx = random($test_size); #ピックアップサイズのndarray
       $test_idx = $test_idx * 9999; # mnistはテストデータが1万個別だてなのでall_data_sizeではない
       $test_idx = convert($test_idx,long); # 整数に

    my $test_X_PDL = $test_x->index1d($test_idx)->sever;
    my $test_T_PDL = $test_t->index1d($test_idx)->sever;

    my $train_acc = $network->accuracy($pickup_X_PDL_T , $pickup_T_PDL_T);
    my $test_acc = $network->accuracy($test_X_PDL->transpose , $test_T_PDL->transpose);

    Logging("$train_acc | $test_acc");

    #ガーベッジコレクション というかシリアライズすると処理速度が低下しない これをしないと、epoch毎に1.5倍の時間がかかる
    $serialize = freeze $network;
    $network = thaw($serialize);

} # for epoch

#学習結果はロードした学習結果から別のプログラムで計測する

  # ハイパーパラメータ等ハッシュにまとめる
  my $hparams = {};
     $hparams->{input_size} = $input_size;
     $hparams->{hidden_size} = $hidden_size;
     $hparams->{output_size} = $output_size;
     $hparams->{waits_init} = $waits_init;
     $hparams->{L2norm} = $L2norm;
     $hparams->{loss_func} = $loss_func;
     $hparams->{activation} = $activation;
     $hparams->{network} = $network;
     $hparams->{all_data_size} = $all_data_size; 
     $hparams->{pickup_size} = $pickup_size; 
     $hparams->{test_size} = $test_size; 
     $hparams->{batch_size} = $batch_size;
     $hparams->{itre} = $itre; 
     $hparams->{epoch} = $epoch;
     $hparams->{learan_rate} = $learn_rate;
     $hparams->{optimizer} = $optimizer;

my $string_hidden = join('_' , @{$hidden_size});

store $hparams , "multilayernet.hparams_U${string_hidden}_b${batch_size}_ep${epoch}_L2norm${L2norm}_pic${pickup_size}_lr${learn_rate}_$activation";



