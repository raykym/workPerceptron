package IdentityWithLoss_layer;

# 恒等関数レイヤーのクラス

use utf8;
binmode 'STDOUT' , ':utf8';

use feature 'say';

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
    $self->{y} = $X; # 恒等関数なのでそのまま
    $self->{loss} = Ml_functions::sum_squared_error($self->{y} , $self->{t});

    #  say "DEBUG: sofrmax: forward";
    #  say $self->{t};
    #  say "";

    return $self->{loss};
}

sub backward {
    my ( $self , $DOUT ) = @_;
    #$DOUT = topdl($DOUT); 
    $DOUT = 1;

    my $DX = $self->{y};

    return $DX;
}


1;


