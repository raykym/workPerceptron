package Affine_layer;

# Affineレイヤーのクラス

use utf8;
binmode 'STDOUT' , ':utf8';

use PDL;
use PDL::Core ':Internal';
use PDL::NiceSlice;

use feature 'say';


sub new {
    my $proto = shift;
    my $class = ref $proto || $proto;

    my $self = {};
       $self->{W} = shift;
       $self->{W} = topdl($self->{W});
       #$self->{W} = $self->{W}->transpose; # PDLの為 -> TwoLayerNetで呼び出されるので転置済み
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
=pod
    say "DEBUG: Affine forward X";
    say $X->shape;
    say "DEBUG: Affine forward self->{w}";
    say $self->{W}->shape();
    say "DEBUG: Affine forward self->{b}";
    say $self->{b}->shape();
    say "";
=cut
    my $OUT = $X x $self->{W} + $self->{b};

    return $OUT;
}

sub backward {
    my ( $self , $DOUT ) = @_;
    #$DOUT = topdl($DOUT); 
    #  say "DEBUG: Affine: backward DOUT";
    #  say $DOUT->shape;

    $self->{W} = $self->{W}->transpose;
    #  say "DEBUG: Affine: backward W_t";
    #  say $self->{W}->shape;
    my $DX = $DOUT x $self->{W};

    $self->{X} = $self->{X}->transpose;
    #  say "DEBUG: Affine: backward X_t";
    #  say $self->{X}->shape;
    $self->{dW} = $self->{X} x $DOUT;

    #  say "Affine backward: DOUT";
    #  say $DOUT->shape;
    #  say "";

    #行方向にsum axis=0
    #$self->{db} = sum($DOUT);
    my $DOUT_tmp = $DOUT->copy;
       $DOUT_tmp = $DOUT_tmp->xchg(0,1); # 列にノードを持ってくる
    $DOUT_tmp = sumover($DOUT_tmp); # 行をsumする
    $self->{db} = $DOUT_tmp;

    #  say "Affine: backward:";
    #  say $self->{db}->shape;

    undef $DOUT_tmp;
    # 転置を戻す
    $self->{W} = $self->{W}->transpose;
    $self->{X} = $self->{X}->transpose;

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


