#!/usr/bin/env perl
#
use strict;
use  warnings;
use utf8;

use v5.32;

binmode 'STDOUT' , ':utf8';

$|=1;

use PDL;
use PDL::IO::FITS;

my $x = rfits('./MNIST/train-images-idx3-ubyte.gz');

say $x->shape;

