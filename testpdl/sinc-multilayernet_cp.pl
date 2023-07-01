#!/usr/bin/env perl
#
# sincを学習して再現性を試す。　普遍性の実験
# tainerクラスを作る
# 速度低下の問題が発生。。。原因がわからない。。。。編集残しで$networkを使っていたため、書き直して完了
# どうにも、問題なさそうだけど警告が最後に出る。->storeコマンドで発生している　データ的には問題はない

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
#use TwoLayerNet;
    # 直接編集して、書き換えが必要！！！！！！！
    # 用途に応じて、Sigmoid_layer , IdentityWithLoss_layerの指定が必要
use MultiLayerNet;
    # last_layerについては直接書き換えが必要
use Adam_optimizer;
#use Trainer;

use Data::Dumper;
use List::Util;

use Storable qw/ freeze thaw store retrieve/;
use PDL::IO::Storable;

# データ取得
my $trainMake = Sincpdl->new;
my ( $all_x , $all_t ) = $trainMake->make;

my $input_size = 2;
my $hidden_size =  [ 100 , 100 ];
my $output_size = 1;
my $activation = 'relu';
my $waits_init = "he";
my $L2norm = 0.1;
my $loss_func = 'mean_squared_error';
#my $network = TwoLayerNet->new($input_size , $hidden_size , $output_size , $waits_init , $L2norm );
my $network = MultiLayerNet->new($input_size , $hidden_size , $output_size , $activation , $waits_init , $L2norm , $loss_func);
    # input_size , hidden_size , output_size , waits_init , weight_decay_rambda
    # 活性化関数はTwoLayerNetで直接指定

my @dims = $all_x->dims;
my $all_data_size = $dims[0]; # Sincpdlの最初の次元が個数になっているので
my $pickup_size = 24000; #データのピックアップ数
my $test_size = 1000; #テストデータのピックアップ数
my $batch_size = 100; #バッチ数
my $itre = $pickup_size / $batch_size ; # イテレーター数
my $epoch = 500; #エポック数

my $learn_rate = 0.001; # optimizerで指定する
my $optimizer = Adam_optimizer->new($learn_rate);

my $serialize = undef;

my $verbose = 1;


# サブ ルーチンエリア
sub Logging {
        my $logline = shift;
        my $dt = time();
        say "$dt | $logline";

        undef $dt;
        undef $logline;

        return;
}

# ローカルパッケージ
my $trainer = Trainer->new($network , $all_x , $all_t , $epoch , $batch_size , $optimizer , $pickup_size , $test_size);
# 上手く学習出来ていない。。。

   $trainer->train;

my $string_hidden = join('_' , @{$hidden_size});

# 学習後に元データを入力して再現性を確認する
   open ( my $fh , '>' , "./sinc_plotdata.txt_U${string_hidden}_b${batch_size}_ep${epoch}_L2norm${L2norm}_pic${pickup_size}_lr${learn_rate}_$activation");
    # x,yを与えて結果をまとめてファイル出力,gnuplotで利用する
    # 意図的に範囲を広げて汎用性を確認する
    for ( my $x = -10 ; $x <= 10 ; $x++  ) {
        for ( my $y = -10 ; $y <= 10 ; $y++  ) {
            my $RET = $trainer->{network}->predict(pdl([ $x , $y ]));  # mainスコープの$networkでは学習結果を利用できない。trainerから学習済みのnetworkにアクセスする必要がある。
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

     store $hparams , "multilayernet.hparams_U${string_hidden}_b${batch_size}_ep${epoch}_L2norm${L2norm}_pic${pickup_size}_lr${learn_rate}_$activation";

     sleep 1;

     exit;



package Trainer;

use v5.32;
use utf8;

use PDL;
use PDL::Core ':Internal';
use PDL::NiceSlice;

use Storable qw(freeze thaw);
use PDL::IO::Storable;

sub new {
    my $proto = shift;
    my $class = ref $proto || $proto;

    my $self = {};

    my ( $network , $all_x , $all_t , $epoch , $batch_size , $optimizer , $pickup_size , $test_size) = @_;

       $self->{network} = $network;
       $self->{all_x} = $all_x;
       $self->{all_t} = $all_t;
       $self->{epoch} = $epoch;
       $self->{batch_size} = $batch_size;
       $self->{optimizer} = $optimizer;
       $self->{pickup_size} = $pickup_size;
    my @dims = $self->{all_x}->dims;
       $self->{all_data_size} = $dims[0]; 
       $self->{test_size} = $test_size; 
       $self->{itre} = $pickup_size / $batch_size ; # イテレーター数

    bless $self , $class;

    undef $network;
    undef $all_x;
    undef $all_t;
    undef $epoch;
    undef $batch_size;
    undef $optimizer;
    undef $pickup_size;
    undef $test_size;

    return $self;
}


sub makeindex {
    my $self = shift;
    #  my $pick_size = shift;
    # train_xからランダムにピックアップする（非復元抽出）為のindex作成 
    # epoch単位、pickup_sizeでデータは一意に成る。　epochを繰り返すと重複は起こり得る
    my @index = ();
    my @array = ();

    for my $i ( 0 .. $self->{all_data_size} -1) { 
        push(@array , [ $i , int(rand($self->{all_data_size} * 10)) ] ); 
    } 
    my @array_sort = sort { $a->[1] <=> $b->[1] } @array; # 乱数側をソートすると、indexがシャッフルされる

    for my $idx ( 0 .. $self->{pickup_size} - 1 ) {
        push ( @index , $array_sort[$idx]->[0]);
    }
    return @index;
}

sub train {
    my $self = shift;

    # 学習の繰り返し回数 
    for my $epoch_cnt ( 1 .. $self->{epoch} ) {
        &::Logging("epoch: $epoch_cnt");

        my $x_batch = null;
        my $t_batch = null;
        my $loss_array = [];

        my @index = $self->makeindex(); # 非復元型抽出 エポック単位ではサンプルの重複はしない
        my $pickup_idx = pdl(@index);
        undef @index;

        my $pickup_X_PDL = $self->{all_x}->index1d($pickup_idx)->sever;
        my $pickup_T_PDL = $self->{all_t}->index1d($pickup_idx)->sever;
    
        # accuraryで利用する転置
        my $pickup_X_PDL_T = $pickup_X_PDL->copy;
           $pickup_X_PDL_T = $pickup_X_PDL->transpose;
        my $pickup_T_PDL_T = $pickup_T_PDL->copy;
           $pickup_T_PDL_T = $pickup_T_PDL->transpose;

        for (my $idx=0 ; $idx <= $self->{itre} -1 ; $idx++ ) {
            # バッチに切り出し
            $x_batch = $pickup_X_PDL->range($idx * $self->{batch_size} , $self->{batch_size})->sever;
            $t_batch = $pickup_T_PDL->range($idx * $self->{batch_size} , $self->{batch_size})->sever;

	    $x_batch = $x_batch->transpose;
            $t_batch = $t_batch->transpose;

            # 傾き計算
	    my $grad = $self->{network}->gradient($x_batch , $t_batch);
	    #my $grad = $self->{network}->numerical_gradient($x_batch , $t_batch);

            # 更新 
            $self->{optimizer}->update($self->{network}->{params} , $grad );

	    push(@{$loss_array} , $self->{network}->loss($x_batch , $t_batch));

        } # for $idx

        # エポック毎にloss表示
        my $avg = undef;
           $avg = List::Util::sum(@{$loss_array});
=pod
        for my $PDL (@{$loss_array} ) {
            #   print "$PDL ";
            my @sum = list($PDL);
               $avg += $sum[0];
        }
=cut
        #say "";
        my @tmp = @{$loss_array};
        my $div = $#tmp + 1;
        $avg /= $div;
        &::Logging("epoch loss: $avg  ($div)");

        undef $div;
        undef @tmp;
        undef $loss_array;


        # testデータの選択 (復元性抽出）
        my $test_idx = random($self->{test_size}); #ピックアップサイズのndarray
           $test_idx = $test_idx * ($self->{all_data_size} - 1); # 最大値は全データ個数
           $test_idx = convert($test_idx,long); # 整数に

        my $test_X_PDL = $self->{all_x}->index1d($test_idx)->sever;
        my $test_T_PDL = $self->{all_t}->index1d($test_idx)->sever;

        my $train_acc = $self->{network}->accuracy($pickup_X_PDL_T , $pickup_T_PDL_T);
        my $test_acc = $self->{network}->accuracy($test_X_PDL->transpose , $test_T_PDL->transpose);
        &::Logging("$train_acc | $test_acc");

        #ガーベッジコレクション というかシリアライズすると処理速度が低下しない epoch毎に1.5倍の時間がかかる
        $serialize = freeze $self->{network};
        $self->{network} = thaw($serialize);

=pod
        undef $test_X_PDL;
        undef $test_T_PDL;
        undef $pickup_X_PDL;
        undef $pickup_T_PDL;
        undef $pickup_X_PDL_T;
        undef $pickup_T_PDL_T;
	undef $x_batch;
	undef $t_batch;
	undef $test_idx;
	undef $pickup_idx;

        $serialize = freeze $self->{all_x};
        $self->{all_x} = thaw($serialize);
        $serialize = freeze $self->{all_t};
        $self->{all_t} = thaw($serialize);
=cut

    } # for epoch 

    return $self;
} 


