#!/usr/bin/env perl
#
# calcsumとReLUの連結
#

use strict;
use warnings;
use utf8;
use feature 'say';

binmode 'STDOUT' , ':utf8';

use FindBin;
use lib "$FindBin::Bin/..";

use Perceptron;

my $perceptron = Perceptron->new;

$perceptron->waitsinit(1);

$perceptron->input([ 100 , 100 ]);

#my $res = $perceptron->calcReLU();

#$perceptron->calcSum();

#my $res = $perceptron->ReLU($perceptron->calcSum());
my $res = $perceptron->calcSum->ReLU();

say $res;
