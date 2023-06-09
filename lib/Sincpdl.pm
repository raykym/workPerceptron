package Sincpdl;

# Sinc関数の-10から10までの間をPDLとしてデータを作成する。
# my $obj = Sincpdl->new;
# ($train_x , $train_l ) = $obj->make;
#

use v5.32;
use utf8;

binmode 'STDOUT' , ':utf8';

use PDL;
use PDL::Core ':Internal';
use PDL::NiceSlice;

sub new {
   my $proto = shift;
   my $class = ref $proto || $proto;

   my $self = {};

   bless $self , $class;

   return $self;
}

sub make {

    my $sample = {};
    for ( my $x = -10 ; $x <= 10 ; $x+=0.1  ) {
        for ( my $y = -10 ; $y <= 10 ; $y+=0.1  ) {
           my $z = undef;
           if ( $x == 0 && $y == 0 ) {
               $z = 0;
           } else {
               $z = (sin ( sqrt( $x**2 + $y**2 ) ) /  sqrt( $x**2 + $y**2 ));
           }
            push (@{$sample->{input}} , [ $x , $y ]);
            push (@{$sample->{class}} , [ $z ]); 

        } # for y
    } # for x

    my $train_x = pdl($sample->{input});
    my $train_t = pdl($sample->{class});
                                       #  個数 , データ
       $train_x = $train_x->transpose; # ( 40401 , 2 ) に変更
       $train_t = $train_t->transpose; # ( 40401 , 1 ) に変更

    return ( $train_x , $train_t);
}

1;

