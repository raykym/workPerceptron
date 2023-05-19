#!/usr/bin/env perl
#
# MnistLoadモジュールを試す
#

use strict;
use warnings;
use utf8;
use feature 'say';
use v5.32;

binmode 'STDOUT' , ':utf8';

$|=1;

use PDL;
use PDL::Core ':Internal';
use PDL::NiceSlice;

use FindBin;
use lib "$FindBin::Bin/lib";
use MnistLoad;

my ( $train_x , $train_l , $test_x , $test_l ) = MnistLoad::mnistload();

#$train_x = topdl($train_x);
#$train_l = topdl($train_l);
#$test_x = topdl($test_x);
#$test_l = topdl($test_l);

say $train_x->shape;
say $train_l->shape;
say $test_x->shape;
say $test_l->shape;

    $train_x = $train_x->reshape(60000,28,28);
    #say $train_x->range(0);
say $train_l(0);

#    $test_x = $test_x->reshape(10000,28,28);
#say $test_x->range(0);
#say $test_l(0);


# hot-one変換
=pod
say $train_l->nelem;
my $T = zeros($train_l->nelem ,10);

say $T->shape;

my $end = $train_l->nelem -1;
for my $i ( 0 .. $end ) {
    $T($i,list($train_l($i))) .= 1;
}
=cut
my $T = MnistLoad::chg_hotone($train_l);

use PDL::NiceSlice;
# サブルーチンを挟むとNiceSliceが解除されたらしい。。。？
say $T(0:9);

# 標準化
    $train_x = $train_x->reshape(60000,28,28);
say $train_x->range(0);

=pod
 my $X = convert( $train_x , double);
    $X /= 255.0;
say $X->range(0);
=cut

say $train_x->info;

#   $train_x = convert($train_x , double);

my $X = MnistLoad::normalize($train_x);
    
say $X->range(0);


