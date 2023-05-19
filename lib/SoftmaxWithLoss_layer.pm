package SoftmaxWithLoss_layer;

# softmaxレイヤーのクラス

use utf8;
binmode 'STDOUT' , ':utf8';

use PDL;
use PDL::Core ':Internal';
use PDL::NiceSlice;

use lib '/home/debian/perlwork/work/workPerceptron/lib';
use Ml_functions;


sub new {
    my $proto = shift;
    my $class = ref $proto || $proto;

    my $self = {};
       $self->{loss} = null;
       $self->{y} = null;
       $self->{t} = null;

    bless $self , $class;

    return $self;
}


sub forward {
    my ($self , $X , $T) = @_;
    $X = topdl($X);
    $T = topdl($T);

    $self->{t} = $T;
    $self->{y} = Ml_functions::softmax($X);
    $self->{loss} = Ml_functions::cross_entropy_error($self->{y} , $self->{t});

    return $self->{loss};
}

sub backward {
    my ( $self , $DOUT ) = @_;
    $DOUT = topdl($DOUT); 
    $DOUT = 1;
    my @dims = $self->{t}->dims;
    my $batch_size = $dims[1]; # PDLでは列を指定する必要があるはず、backwardの転置は未確認

    my $DX .= ( $self->{y} - $self->{t} ) / $batch_size;

    return $DX;
}


1;


