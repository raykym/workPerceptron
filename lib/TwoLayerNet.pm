package TwoLayerNet;

use v5.32;

# numPyとPDLの仕様の違い (行、列）と（列、行）については 入力前に転置する、waitsは転置しておく
# 設定項目は基本的にpythonの記述に準じて、後に修正を行う方針

use utf8;
binmode 'STDOUT' , ':utf8';

use Carp;

use PDL;
use PDL::Core ':Internal';
use PDL::NiceSlice;
use PDL::GSL::RNG;

use FindBin;
use lib "$FindBin::Bin/../lib";
use Ml_functions;
#
use Relu_layer;
use Affine_layer;
use SoftmaxWithLoss_layer;
use Sigmoid_layer;
use IdentityWithLoss_layer;


use Tie::IxHash;
use Time::HiRes qw / time /;



sub new {
    my $proto = shift;
    my $class = ref $proto || $proto;
    my ( $input_size , $hidden_size , $output_size , $weight_init_std , $weight_decay_rambda ) = @_;
       $weight_init_std = 0.01 if ! defined $weight_init_std;
       # xavier , heを入れるとそれぞれの初期化になる

    my $rng = PDL::GSL::RNG->new('mt19937_1999');
       $rng->set_seed(time());

    my $self = {};
       # W1のweight調整
       if ($weight_init_std =~ /^[0-9]+$|^[0-9]+\.[0-9]+$/ ) {
           # 数値ならスルー
       } elsif ($weight_init_std eq 'xavier' ) {
           $weight_init_std = sqrt( 1 / $input_size );
       } elsif ( $weight_init_std eq 'he' ) {
           $weight_init_std = sqrt( 2 / $input_size );
       }
       $self->{params}->{W1} = $weight_init_std * $rng->ran_gaussian($weight_init_std ,$input_size , $hidden_size );
       # ($hidden_size , $input_size)に転置する
       $self->{params}->{W1} = $self->{params}->{W1}->transpose; # waitsは転置する
       $self->{params}->{b1} = zeros($hidden_size);
       # W2のweight調整
       if ($weight_init_std =~ /^[0-9]+$|^[0-9]+\.[0-9]+$/ ) {
           # 数値ならスルー
       } elsif ($weight_init_std eq 'xavier' ) {
           $weight_init_std = sqrt( 1 / $hidden_size );
       } elsif ( $weight_init_std eq 'he' ) {
           $weight_init_std = sqrt( 2 / $hidden_size );
       }
       $self->{params}->{W2} = $weight_init_std * $rng->ran_gaussian($weight_init_std ,$hidden_size , $output_size );
       # ($output_size , $hidden_size)に転置する
       $self->{params}->{W2} = $self->{params}->{W2}->transpose; 
       $self->{params}->{b2} = zeros($output_size);

       # レイヤの生成
       tie( my %layers , 'Tie::IxHash' );  # OrderdDictをIxHashで置き換え
       $self->{layers} = \%layers;

       $self->{layers}->{Affine1} = Affine_layer->new($self->{params}->{W1} , $self->{params}->{b1});
       #$self->{layers}->{Relu1} = Relu_layer->new();
       $self->{layers}->{Sigmoid1} = Sigmoid_layer->new();
       $self->{layers}->{Affine2} = Affine_layer->new($self->{params}->{W2} , $self->{params}->{b2});
       #$self->{lastLayer} = SoftmaxWithLoss_layer->new();
       $self->{lastLayer} = IdentityWithLoss_layer->new();

       # 追加
       $self->{weight_decay_rambda} = $weight_decay_rambda;
       $self->{weight_decay_rambda} = 0 if (! defined($self->{weight_decat_rambda}));

     bless $self , $class;

     return $self;
}

sub predict {
    my $self = shift;
    my $X = shift;
    #$X = topdl($X);

    for my $key ( keys %{$self->{layers}} ) {
	#say "DEBUG: predict: key: $key";
	$X = $self->{layers}->{$key}->forward($X); 
    }
    return $X;
}

sub loss {
    my ($self , $X , $T ) = @_;

    my $Y = $self->predict($X);

    # weight_decayを追加する Affine　2層のweightから算出する
    my $weight_decay = 0;
       $weight_decay += 0.5 * $self->{weight_decay_rambda} * sum($self->{params}->{W1} ** 2);
       $weight_decay += 0.5 * $self->{weight_decay_rambda} * sum($self->{params}->{W2} ** 2);

    return $self->{lastLayer}->forward($Y , $T) + $weight_decay;
}

sub accuracy {
    my ( $self , $X , $T ) = @_;

    my $Y = $self->predict($X);
    my $Yi = Ml_functions::argmax($Y);
    my $Ti = Ml_functions::argmax($T) if ( $T->ndims != 1 ) ;
    #my $accuracy = sum($Y == $T ) / float($X->shape(0)); # 直訳
    my @shape = $X->dims; #(列、行) 行を指定する （データ個数のはず)
    my $accuracy = sum($Yi == $Ti ) / float($shape[1]); # 1次元のインデックスが一致する場所を足し合わせて、データ数で割る 
    #my $accuracy = sum($Yi == $Ti ) / double($shape[1]);  

    undef @shape;
    undef $Y;
    undef $Yi;
    undef $Ti;

    return $accuracy;
}

sub numerical_gradient {
    my ( $self , $X , $T ) = @_;
=pod
    say "DEBUG: numerical_gradient: X";
    say $X->shape;
    say "DEBUG: numerical_gradient: T";
    say $T->shape;
    say "";
=cut

    my $loss_W = sub {
	    #  my ( $X , $T ) = @_; # 無記名関数をカプセル化しない事
	    #   $X = topdl($X);
=pod
		     say "DEBUG: loss_W :X";
		     say $X->shape;
		     say "DEBUG: loss_W: T";
		     say $T->shape;
		     say "";
=cut
		     $self->loss($X,$T);

                     };

    my $grads = {};
       $grads->{W1} = Ml_functions::numerical_gradient($loss_W , $self->{params}->{W1} );
       $grads->{b1} = Ml_functions::numerical_gradient($loss_W , $self->{params}->{b1} );
       $grads->{W2} = Ml_functions::numerical_gradient($loss_W , $self->{params}->{W2} );
       $grads->{b2} = Ml_functions::numerical_gradient($loss_W , $self->{params}->{b2} );

    undef $loss_W; 

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
	       #  say "gradient: dout: while in $key";
	       #  say $dout->shape;

           $dout = $self->{layers}->{$key}->backward($dout);
       }
       undef @keylist; 

    my $grads = {};     # weight_decayを追加した
    $grads->{W1} = $self->{layers}->{Affine1}->dW() + $self->{weight_decay_lambda} * $self->{params}->{W1};
    $grads->{b1} = $self->{layers}->{Affine1}->db();
    $grads->{W2} = $self->{layers}->{Affine2}->dW() + $self->{weight_decay_lambda} * $self->{params}->{W2};
    $grads->{b2} = $self->{layers}->{Affine2}->db();

    return $grads;
}



1;



