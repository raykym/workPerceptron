#!/usr/bin/env perl

use strict;
use warnings;
use utf8;
use feature 'say';
use v5.32;

binmode 'STDOUT' , ':utf8';

$|=1;

use Archive::Zip qw( :ERROR_CODES :CONSTANTS );

my $MNIST = 

# Zipではなく、tar.gzだったので取りやめ
