package Trainer;

# MNIST用のtrainerモジュール　他でも使えるかもしれない
# ゼロから作るdeeplearningとは処理の流れが少し違う
# パラメーターも増えている

# PDLはパッケージや関数に引き渡すとき、PDL::Coreだけを引き継ぐので、
# 必要なPDLモジュールは引き継ぎ先のパッケージ、関数内で再度useする必要がある。

use v5.32;
use utf8;
binmode 'STDOUT' , ':utf8';
use Carp;

use PDL;
use PDL::Core ':Internal';
use PDL::NiceSlice;
use Storable qw/ freeze thaw store /;
use PDL::IO::Storable;

use PDL::Slices;

sub new {
    my $proto = shift;
    my $class = ref $proto || $proto;
    my ($network , $train_x , $train_t , $test_x , $test_t , $epoch , $batch_size , $optimizer , $pickup_size , $test_size , $verbose ) = @_;

       if ( ! defined $verbose ) {
           $verbose = 1; # true
       }  

    my $self = {};
    bless $self , $class;

       $self->{network} = $network;
       $self->{train_x} = $train_x;
       $self->{train_t} = $train_t;
       $self->{test_x} = $test_x;
       $self->{test_t} = $test_t;
       $self->{epoch} = $epoch;
       $self->{batch_size} = $batch_size;
       $self->{optimizer} = $optimizer;
       $self->{pickup_size} = $pickup_size;
       $self->{test_size} = $test_size;
       my @dims = $train_x->dims;
       $self->{all_data_size} = $dims[0];
       $self->{verbose} = $verbose;
       $self->{itre} = $pickup_size / $batch_size;

       undef @dims;

    return $self;

}

sub makeindex {
    my $self = shift;
    my $pickup_size = $self->{pickup_size};
    my $all_data_size = $self->{all_data_size};

    #&::Logging("DEBUG: all_data_size: $all_data_size");
    #&::Logging("DEBUG: pickup_size: $pickup_size");

    # train_xからランダムにピックアップする（非復元抽出）為のindex作成 
    # epoch単位、pickup_sizeでデータは一意に成る。　epochを繰り返すと重複は起こり得る
    my @index = ();
    my @array = ();

    for my $i ( 0 .. $all_data_size -1) { 
        push(@array , [ $i , int(rand($all_data_size * 10)) ] ); # 乱数は$all_data_sizeよりも大きい数値
    }
    my @array_sort = sort { $a->[1] <=> $b->[1] } @array; # 乱数側をソートすると、indexがシャッフルされる

    for my $idx ( 0 .. $pickup_size - 1 ) {
        push ( @index , $array_sort[$idx]->[0]);
    }

    undef @array;
    undef @array_sort;

    #&::Logging("DEBUG: index: @index");

    return @index;
}

sub train {
    my $self = shift;

    my $serialize = undef;
    my $epoch = $self->{epoch};
    my $train_x = $self->{train_x};
    my $train_t = $self->{train_t};
    my $itre = $self->{itre};
    my $batch_size = $self->{batch_size};
    my $test_size = $self->{test_size};
    my $test_x = $self->{test_x};
    my $test_t = $self->{test_t};

# 学習の繰り返し回数 
for my $epoch_cnt ( 1 .. $epoch ) {
    &::Logging("epoch: $epoch_cnt");

    my $x_batch = null;
    my $t_batch = null;

    my $loss_array = [];

    my @index = $self->makeindex(); # 非復元型抽出 エポック単位ではサンプルの重複はしない
    #my $pickup_idx = pdl(@index);
    #undef @index;

    my $pickup_X_PDL = $train_x->index1d(pdl(@index))->sever;
    my $pickup_T_PDL = $train_t->index1d(pdl(@index))->sever;
    #my $pickup_X_PDL = $train_x->index1d($pickup_idx)->sever;
    #my $pickup_T_PDL = $train_t->index1d($pickup_idx)->sever;

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
        my $grad = $self->{network}->gradient($x_batch , $t_batch);
        #my $grad = $network->numerical_gradient($x_batch , $t_batch);

        # 更新 
        $self->{optimizer}->update($self->{network}->{params} , $grad );

	push(@{$loss_array} , $self->{network}->loss($x_batch , $t_batch)); #バッチ単位のlossが　1 epoch分集まる

    } # for idx

    # エポック毎にloss表示
    my $avg = undef;
    for my $PDL (@{$loss_array} ) {
	my @sum = list($PDL);
           $avg += $sum[0]; 
    }
    my @tmp = @{$loss_array};
    $avg /= $#tmp + 1;
    &::Logging("epoch loss: $avg");

    undef @tmp;
    undef $loss_array;


   # testデータの選択 (復元性抽出）
    my $test_idx = random($test_size); #ピックアップサイズのndarray
       $test_idx = $test_idx * 9999; # mnistはテストデータが1万個別だてなのでall_data_sizeではない
       $test_idx = convert($test_idx,long); # 整数に

    my $test_X_PDL = $test_x->index1d($test_idx)->sever;
    my $test_T_PDL = $test_t->index1d($test_idx)->sever;

    my $train_acc = $self->{network}->accuracy($pickup_X_PDL_T , $pickup_T_PDL_T);
    my $test_acc = $self->{network}->accuracy($test_X_PDL->transpose , $test_T_PDL->transpose);

    &::Logging("$train_acc | $test_acc");

    #ガーベッジコレクション というかシリアライズすると処理速度が低下しない これをしないと、epoch毎に1.5倍の時間がかかる
    $serialize = freeze $self->{network};
    $self->{network} = thaw($serialize);

} # for epoch

} # sub train

1;
