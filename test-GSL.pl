#!/usr/bin/env perl
#
use strict;
use warnings;
use utf8;

use feature 'say';

binmode 'STDOUT' , ':utf8';

use Math::GSL::RNG;
use Math::GSL::Randist qw /gsl_ran_gaussian /;

my $rng = Math::GSL::RNG->new();

#my @array = $rng->get(100);
#say $_ for @array;


my $g_rand = gsl_ran_gaussian($rng->raw(), 1/sqrt(2));

say $g_rand;


