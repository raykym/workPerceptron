#!/usr/bin/env perl
#
# MNSITをMultiLayerNetで学習を試す
# Trainerをファイル切り出して、使うとPDLがメソッドを失う？？？
# パッケージにPDLを引き継いだ場合、core以外は引き継がれないので、必要なメソッドのパッケージを、引受先のパッケージで個別にuseする必要がある。

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
use Trainer;

# MNISTファイルのロード
my ($train_x , $train_t , $test_x , $test_t ) = MnistLoad::mnistload();

# 標準化
$train_x = MnistLoad::normalize($train_x);
$test_x = MnistLoad::normalize($test_x);

#ラベルをhot-one化
$train_t = MnistLoad::chg_hotone($train_t);
$test_t = MnistLoad::chg_hotone($test_t);


my @dims_x = $train_x->dims;
my @dims_t = $train_t->dims;

# パラメータ
my $input_size = $dims_x[1]; #入力データの長さ numpyとは逆
my $hidden_size =  [ 100 , 100 , 100 ];
my $output_size = $dims_t[1]; #出力データの長さ
my $activation = 'relu';   # relu or sigmoid
my $waits_init = "he";     # xavier or he
my $L2norm = 0.9;
my $loss_func = 'cross_entropy_error';  # mean_squared_error(恒等関数) or cross_entropy_error(softmax関数)
my $network = MultiLayerNet->new($input_size , $hidden_size , $output_size , $activation , $waits_init , $L2norm , $loss_func);

my $all_data_size = $dims_x[0]; # Mnistloadの最初の次元が個数になっているので
my $pickup_size = 20000; #データのピックアップ数
my $test_size = 10000; #テストデータのピックアップ数
my $batch_size = 2000; #バッチ数
my $itre = $pickup_size / $batch_size ; # イテレーター数
my $epoch = 100; #エポック数

my $learn_rate = 0.001; # optimizerで指定する
my $optimizer = Adam_optimizer->new($learn_rate);

#my $serialize = undef;
my $verbose = 1;

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

my $trainer = Trainer->new( $network , $train_x , $train_t , $test_x , $test_t , $epoch , $batch_size , $optimizer , $pickup_size , $test_size , $verbose );

   $trainer->train;




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



