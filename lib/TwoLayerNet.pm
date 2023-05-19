package TwoLayerNet;

# numPyとPDLの仕様の違い (行、列）と（列、行）については 入力前に転置する、waitsは転置しておく
# 設定項目は基本的にpythonの記述に準じて、後に修正を行う方針

use utf8;
binmode 'STDOUT' , ':utf8';

use PDL;
use PDL::Core ':Internal';
use PDL::NiceSlice;

use FindBin;
use lib "$FindBin::Bin/../lib";
use Multilayer_PDL;
use Ml_functions;
#
use Relu_layer;
use Affine_layer;
use SoftmaxWithLoss_layer;


use Tie::IxHash;
use Time::HiRes qw / time /;



sub new {
    my $proto = shift;
    my $class = ref $proto || $proto;
    my ( $input_size , $hidden_size , $output_size , $weight_init_std ) = @_;
       $weight_init_std = 0.01 if ! defined $weight_init_std;

    my $rng = PDL::GSL::RNG->new('mt19937_1999');
       $rng->set_seed(time());

    my $self = {};
       $self->{params}->{W1} = $weight_init_std * $rng->ran_gaussian(1 ,$input_size , $hidden_size );
       $self->{params}->{W1} = $self->{params}->{W1}->transpose; # waitsは転置する
       $self->{params}->{b1} = zeros($hidden_size);
       $self->{params}->{W2} = $weight_init_std * $rng->ran_gaussian(1 ,$hidden_size , $output_size );
       $self->{params}->{W2} = $self->{params}->{W2}->transpose; 
       $self->{params}->{b2} = zeros($output_size);

       # レイヤの生成
       tie( my %layers , 'Tie::IxHash' );  # OrderdDictをIxHashで置き換え
       $self->{layers} = \%layers;

       $self->{layers}->{Affine1} = Affine_layer->new($self->{params}->{W1} , $self->{params}->{b1});
       $self->{layers}->{Relu1} = Relu_layer->new();
       $self->{layers}->{Affine2} = Affine_layer->new($self->{params}->{W2} , $self->{params}->{b2});
       $self->{lastLayer} = SoftmaxWithLoss->new();

     bless $self , $class;

     return $self;
}

sub predict {
    my $self = shift;
    my $X = shift;
    $X = topdl($X);

    for my $key ( keys %{$self->{layers}} ) {
       $X .= $self->{layers}->{$key}->forward($X); 
    }
    return $X;
}

sub loss {
    my ($self , $X , $T ) = @_;

    my $Y = $self->predict($X);

    return $self->{layers}->{latLayer}->forward($Y , $T);
}

sub accuracy {
    my ( $self , $X , $T ) = @_;

    my $Y = $self->predict($X);
    $Y .= Ml_functions::argmax($Y);

    $T .= Ml_functions::argmax($T) if ( $T->ndims != 1 ) ;

    #my $accuracy = sum($Y == $T ) / float($X->shape(0)); # 直訳
    my @shape = $X->dims; #(列、行) 行を指定する
    my $accuracy = sum($Y == $T ) / float($shape[1]);  # $X->ndimsで良いのかも？

    undef @shape;
    undef $Y;
    undef $T;

    return $accuracy;
}

sub numerical_gradient {
    my ( $self , $X , $T ) = @_;

    my $loss_W = sub {
                     my ( $x , $T ) = @_;
		     $x = topdl($x);

                     $x = $x->transpose; #何故が転置が解除されるので
                     state $X = $x; # 初期値で固定する
                     
		     $self->loss($X,$T);

                     };

    my $grads = {};
       $grads->{W1} = Ml_functions::numerical_gradient($loss_W , $self->{params}->{W1} );
       # 2回めのときに、$loss_Wの引数に{b1}が入ってしまうのでstateで固定
       $grads->{b1} = Ml_functions::numerical_gradient($loss_W , $self->{params}->{b1} );
       $grads->{W2} = Ml_functions::numerical_gradient($loss_W , $self->{params}->{W2} );
       $grads->{b2} = Ml_functions::numerical_gradient($loss_W , $self->{params}->{b2} );

    undef $loss_W; # 呼び出し毎にstateをリセットする

    return $grads;

}

sub gradient {
    my ( $self , $X , $T ) = @_;
    $self->loss($X , $T);

    my $dout = 1;
       $dout = $self->{lastLayer}->backward($dout);

       my @keylist; #{layers}のキーを逆順に実行する(IxHash)
       for my $key ( keys %{$self->{layers}} ) {
           push(@keylist , $key);
       }
       while ( my $key = pop @keylist ) {
           $dout = $self->{layers}->{$key}->backward($dout);
       }
       undef @keylist; 

    my $grads = {};
    $grads->{W1} = $self->{layers}->{Affine1}->dW();
    $grads->{b1} = $self->{layers}->{Affine1}->db();
    $grads->{W2} = $self->{layers}->{Affine2}->dW();
    $grads->{b2} = $self->{layers}->{Affine2}->db();

    return $grads;
}



1;



