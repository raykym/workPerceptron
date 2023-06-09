package Relu_layer;

# Reluレイヤーのクラス

use utf8;
binmode 'STDOUT' , ':utf8';

use PDL;
use PDL::Core ':Internal';
use PDL::NiceSlice;


sub new {
    my $proto = shift;
    my $class = ref $proto || $proto;

    my $self = {};
       $self->{mask} = undef;
       # maskは比較して0以下のノードをTrue("1")とするndarrayが必要

    bless $self , $class;

    return $self;
}

# mask用関数
sub relumask {
    my ( $self , $X ) = @_;
    $X = topdl($X);

    my $Z = zeros($X);

    return $Z >= $X;  # $Xの0以下のノードが1になるインデックスが返る
}

sub maskfilter {
    my ($self , $X ) = @_; 
    #maskを行列に適用する

    my $Z = zeros($self->{mask});
    my $FILTER = $Z == $self->{mask}; # maskの0,1がはんてんしたFILTER

    my $OUT = $X * $FILTER; # maskの位置が0に成る

    undef $Z;
    undef $FILTER;
    
    return $OUT;
}

sub forward {
    my ($self , $X ) = @_;
    $X = topdl($X);

    $self->{mask} = $self->relumask($X);

    my $OUT = $X->copy;

    #$OUT($self->{mask}) = 0; # 直訳だとmaskの位置を0にする
    # PDLだと機能が無いのでmaskfilterを作成
    # relumaskに加えると論理がわかりにくく成るのであえて別にする

    $OUT = $self->maskfilter($OUT);

    return $OUT;
}

sub backward {
    my ( $self , $DOUT ) = @_;
    $DOUT = topdl($DOUT); 

    $DOUT = $self->maskfilter($DOUT);

    my $DX = $DOUT;

    return $DX;
}


1;


