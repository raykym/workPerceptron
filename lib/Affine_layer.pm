package Affine_layer;

# Affineレイヤーのクラス

use utf8;
binmode 'STDOUT' , ':utf8';

use PDL;
use PDL::Core ':Internal';
use PDL::NiceSlice;


sub new {
    my $proto = shift;
    my $class = ref $proto || $proto;

    my $self = {};
       $self->{W} = shift;
       $self->{W} = topdl($self->{W});
       $self->{W} = $self->{W}->transpose; # PDLの為
       $self->{b} = shift;
       $self->{b} = topdl($self->{b});

       $self->{X} = null;
       $self->{dW} = null;
       $self->{db} = null;

    bless $self , $class;

    return $self;
}


sub forward {
    my ($self , $X ) = @_;
    $X = topdl($X);

    $self->{X} = $X;

    my $OUT = $X x $self->{W} + $self->{b};

    return $OUT;
}

sub backward {
    my ( $self , $DOUT ) = @_;
    $DOUT = topdl($DOUT); 
    # 逆伝搬の場合転置が再度入るかもしれない!!!!!!

    $self->{W} = $self->{W}->transpose;
    my $DX = $DOUT * $self->{W};

    $self->{X} = $self->{X}->transpose;
    $self->{dW} = $self->{X} x $DOUT;
    $self->{db} = sum($DOUT);

    return $DX;
}

sub dW {
    my $self = shift;
    # gradのゲッター(waits)

    return $self->{dW};
}

sub db {
    my $self = shift;
    # gradのゲッター(bias)

    return $self->{db};
}

1;


