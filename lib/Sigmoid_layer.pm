package Sigmoid_layer;

# Sigmoidレイヤーのクラス

use utf8;
binmode 'STDOUT' , ':utf8';

use PDL;
use PDL::Core ':Internal';
use PDL::NiceSlice;


sub new {
    my $proto = shift;
    my $class = ref $proto || $proto;

    my $self = {};
       $self->{out} = undef;

    bless $self , $class;

    return $self;
}


sub forward {
    my ($self , $X ) = @_;
    $X = topdl($X);

    my $OUT = 1 / ( 1 + exp(-$X));

    $self->{out} = $OUT;

    return $OUT;
}

sub backward {
    my ( $self , $DOUT ) = @_;
    $DOUT = topdl($DOUT); 

    my $DX = $DOUT * ( 1.0 - $self->{out} ) * $self->{out};;

    return $DX;
}


1;


