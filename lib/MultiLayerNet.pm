package MultiLayerNet;

# ゼロから作るDeepLearningのMultilayerNetクラスを書き換え
# perlならもう少し別の書き方をするだろうけど、原文の意図に沿うようにしている

use v5.32;
use utf8;
use Carp;

binmode 'STDOUT' , ':utf8';

use PDL;
use PDL::Core ':Internal';
use PDL::NiceSlice;
use PDL::GSL::RNG;
use Tie::IxHash;
use Time::HiRes qw/ time /;

use FindBin; # 実行はworkPerceptron配下のディレクトリから呼び出されることを想定
use lib "$FindBin::Bin/../lib";
use Ml_functions;
use Relu_layer;
use Affine_layer;
use SoftmaxWithLoss_layer;
use Sigmoid_layer;
use IdentityWithLoss_layer;

sub new {
    my $proto = shift;
    my $class = ref $proto || $proto;

    my ( $input_size , $hidden_size_list , $output_size , $activation , $weight_init_std , $weight_decay_lambda ) = @_;

    my $self = {};
       $self->{input_size} = $input_size;
       $self->{output_size} = $output_size;
       $self->{hidden_size_list} = $hidden_size_list; # ARRAYref
       my @tmp = @{$self->{hidden_size_list}};
       $self->{hidden_layer_num} = $#tmp + 1; # 添字最大値+1 len関数の出力相当
       undef @tmp;
       $self->{weight_init_std} = $weight_init_std; #数値 or sigmoid,relu,xavier.he
       $self->{weight_decay_lambda} = $weight_decay_lambda;  # L2norm
       $self->{params} = {};

       bless $self , $class;

       # 重みの初期化
       $self->_init_weight($weight_init_std);

       # レイヤーの生成
       my $activation_layer = { sigmoid => 'Sigmoid_layer' , relu => 'Relu_layer' };
       tie( my %layers , 'Tie::IxHash' );  # OrderdDictをIxHashで置き換え
       $self->{layers} = \%layers; 
       for my $idx ( 1 .. $self->{hidden_layer_num} ) { 
           my $str = "Affine$idx";
           $self->{layers}->{"Affine$idx"} = Affine_layer->new($self->{params}->{"W$idx"} , $self->{params}->{"b$idx"});
           $self->{layers}->{"Activation_function$idx"} = $activation_layer->{$activation}->new;
       } # for
  # 1層では動作が合わないので、外してみる
       my $num = $self->{hidden_layer_num} + 1; # perlではidxがスコープを外れるので
       $self->{layers}->{"Affine$num"} = Affine_layer->new($self->{params}->{"W$num"} , $self->{params}->{"b$num"});

       #$self->{last_layer} = SoftmaxWithLoss_layer->new();
       $self->{last_layer} = IdentityWithLoss_layer->new();

    return $self;
}

sub _init_weight {
    my ($self , $weight_init_std ) = @_;

    my $rng = PDL::GSL::RNG->new('mt19937_1999');
       $rng->set_seed(time());

    my $all_size_list = [ $self->{input_size} , @{$self->{hidden_size_list}} , $self->{output_size} ] ; # hidden_size_list先頭と末尾にinput_sizeとoutput_sizeを追加したarrayRef
    my @tmp = @{$all_size_list};
    for my $idx ( 1 .. $#tmp +1 ) {
           # 対応する添字に使うので1スタートに成っている
           # 初期化は一つ前の層を利用するので-1してもall_size_listの添字と一致する
        my $scale = $self->{weight_init_std};
        if ($self->{weight_init_std} =~ 'relu' || $self->{weight_init_std} =~ 'he') {
            $scale = sqrt( 2 / $all_size_list->[$idx -1] );
        } elsif ( $self->{weight_init_std} =~ 'sigmoid' || $self->{weight_init_std} =~ 'xavier' ) {
            $scale = sqrt( 1 / $all_size_list->[$idx -1] );
        } # if

        $self->{params}->{"W$idx"} = $rng->ran_gaussian($scale ,$all_size_list->[$idx -1] , $all_size_list->[$idx] );
        $self->{params}->{"W$idx"} = $self->{params}->{"W$idx"}->transpose; # waitsは転置する
        $self->{params}->{"b$idx"} = zeros($all_size_list->[$idx]);

    } # for idx

}

sub predict {
    my ($self , $X) = @_;

    for my $key (keys %{$self->{layers}} ) {
	    #&::Logging("DEBUG: predict: key: $key ");
        $X = $self->{layers}->{$key}->forward($X);
    } 
    #my @tmp = $X->dims;
    #&::Logging("DEBUG: predict: X: @tmp ");

    return $X;
}

sub loss {
    my ($self , $X , $T) = @_;

    #my @tmp = $X->dims;
    #&::Logging("DEBUG: loss: X: @tmp ");

    my $Y = $self->predict($X);

    #my @tmp2 = $Y->dims;
    #&::Logging("DEBUG: loss: Y: @tmp2 ");

    my $weight_decay = 0;
    for my $idx ( 1 .. $self->{hidden_layer_num} + 1 ) {
	    # my $W = $self->{params}->{"W$idx"};
	    # $weight_decay += 0.5 * $self->{weight_decay_lambda} * sum($W ** 2);
        $weight_decay += 0.5 * $self->{weight_decay_lambda} * sum($self->{params}->{"W$idx"} ** 2);
    }
    return $self->{last_layer}->forward($Y,$T) + $weight_decay;
}

sub accuracy {
    my ($self , $X , $T) = @_;
    my $Y = $self->predict($X);
    my $Yi = Ml_functions::argmax($Y);
    my $Ti = Ml_functions::argmax($T) if ($T->ndims != 1);
    my @dims = $X->dims;
    my $accuracy = sum($Yi == $Ti) / float($dims[1]);

    undef $Yi;
    undef $Ti;
    undef $Y;
    undef @dims;

    return $accuracy;
}

sub numerical_gradient {
    my ($self , $X , $T) = @_;

    my $loss_W = sub {
                $self->loss($X,$T);
            };

    my $grad = {};
    for my $idx ( 1 .. $self->{hidden_layer_num} +2) { # 初期設定のまま、上手く動かないはず
        $grad->{"W$idx"} = Ml_functions::numerical_gradient($loss_W , $self->{params}->{"W$idx"});
        $grad->{"b$idx"} = ML_functions::numerical_gradient($loss_W , $self->{params}->{"b$idx"});
    }
    return $grad;
}

sub gradient {
    my ($self , $X , $T) = @_;

    $self->loss($X , $T);

    #my @tmp = $X->dims;
    #&::Logging("DEBUG: gradient: X: @tmp");

    my $dout = 1;
       $dout = $self->{last_layer}->backward($dout);

    my @layers = values %{$self->{layers}};
    @layers = reverse @layers; #逆順に並び替える
    for my $layer ( @layers ) {
        $dout = $layer->backward($dout);
    }
    my $grad = {};
    for my $idx ( 1 .. $self->{hidden_layer_num} + 1) {
        $grad->{"W$idx"} = $self->{layers}->{"Affine$idx"}->dW + $self->{weight_decay_lambda} * $self->{layers}->{"Affine$idx"}->{W};
        $grad->{"b$idx"} = $self->{layers}->{"Affine$idx"}->db;

    } # for idx
    return $grad;
}

1;
