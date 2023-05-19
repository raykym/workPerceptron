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
use lib "$FindBin::Bin/../lib";
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
say $train_x->range(0);
say $train_l(0);

    $test_x = $test_x->reshape(10000,28,28);
say $test_x->range(0);
say $test_l(0);

    $train_x = $train_x->reshape(60000,784);
    $test_x = $test_x->reshape(60000,1);

    say $train_x->shape;
    say $train_x->range(0);

    say $test_x->shape;
    say $test_x->range(0);


