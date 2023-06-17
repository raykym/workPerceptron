#!/usr/bin/env perl
#
# sincを学習して再現性を試す。　普遍性の実験
#

use v5.32;
use utf8;

binmode 'STDOUT' , ':utf8';
$|=1;

use PDL;
use PDL::Core ':Internal';
use PDL::NiceSlice;

use Time::HiRes '/ time /';

use FindBin;
use lib "$FindBin::Bin/../lib";
use Sincpdl;
use MultiLayerNet;
use Adam_optimizer;

use Data::Dumper;

use Storable qw/ freeze thaw store retrieve/;
use PDL::IO::Storable;

# データ取得
my $trainMake = Sincpdl->new;
my ( $all_x , $all_t ) = $trainMake->make;

my $input_size = 2;
my $hidden_size =  [ 100 , 100 ];
my $output_size = 1;
my $activation = 'relu';   # relu or sigmoid
my $waits_init = "he";     # xavier or he
my $L2norm = 0.0;
my $loss_func = 'mean_squared_error';  # mean_squared_error or cross_entropy_error
my $network = MultiLayerNet->new($input_size , $hidden_size , $output_size , $activation , $waits_init , $L2norm , $loss_func);

my @dims = $all_x->dims;
my $all_data_size = $dims[0]; # Sincpdlの最初の次元が個数になっているので
my $pickup_size = 24000; #データのピックアップ数
my $test_size = 1000; #テストデータのピックアップ数
my $batch_size = 50; #バッチ数
my $itre = $pickup_size / $batch_size ; # イテレーター数
my $epoch = 500; #エポック数

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

#say Dumper $network;
#exit;

# 学習の繰り返し回数 
for my $epoch_cnt ( 1 .. $epoch ) {
    Logging("epoch: $epoch_cnt");

    my $x_batch = null;
    my $t_batch = null;

    my $loss_array = [];

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
        $t_batch = $t_batch->transpose;
	#
	#my @tmp = $x_batch->dims;
	#&::Logging("DEBUG: x_batch: @tmp");

        # 傾き計算
	my $grad = $network->gradient($x_batch , $t_batch);
	#my $grad = $network->numerical_gradient($x_batch , $t_batch);

        # 更新 
        $optimizer->update($network->{params} , $grad );

        push(@{$loss_array} , $network->loss($x_batch , $t_batch)); #バッチ単位のlossが　1 epoch分集まる
	   #  say Dumper $network->{params};

        #  say "itre: $idx loss: $loss ";

    } # for $idx

    # エポック毎にloss表示
    my $avg = undef;
    for my $PDL (@{$loss_array} ) {
	    #	print "$PDL ";
        my @sum = list($PDL);
           $avg += $sum[0];
    }
    #say "";
    my @tmp = @{$loss_array};
    my $div = $#tmp + 1;
    $avg /= $div;
    Logging("epoch loss: $avg  ($div)");

    undef $div;
    undef @tmp;
    undef $loss_array;

    
    # testデータの選択 (復元性抽出）
    my $test_idx = random($test_size); #ピックアップサイズのndarray
       $test_idx = $test_idx * ($all_data_size - 1); # 最大値は全データ個数
       $test_idx = convert($test_idx,long); # 整数に

    my $test_X_PDL = $all_x->index1d($test_idx)->sever;
    my $test_T_PDL = $all_t->index1d($test_idx)->sever;

    my $train_acc = $network->accuracy($pickup_X_PDL_T , $pickup_T_PDL_T);
    my $test_acc = $network->accuracy($test_X_PDL->transpose , $test_T_PDL->transpose);
    Logging("$train_acc | $test_acc");

    #ガーベッジコレクション というかシリアライズすると処理速度が低下しない これをしないと、epoch毎に1.5倍の時間がかかる
    $serialize = freeze $network;
    $network = thaw($serialize);

} # for epoch




my $string_hidden = join('_' , @{$hidden_size});

# 学習後に元データを入力して再現性を確認する
   open ( my $fh , '>' , "./sinc_plotdata.txt_U${string_hidden}_b${batch_size}_ep${epoch}_L2norm${L2norm}_pic${pickup_size}_lr${learn_rate}_$activation");
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

  # イメージ取得 ハッシュにまとめる
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

     store $hparams , "multilayernet.hparams_U${string_hidden}_b${batch_size}_ep${epoch}_L2norm${L2norm}_pic${pickup_size}_lr${learn_rate}_$activation";

