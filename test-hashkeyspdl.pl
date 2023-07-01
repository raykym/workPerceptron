#!/usr/bin/env perl

# hashをkeysで返して、PDLにするテスト

use v5.32;
use utf8;
binmode 'STDOUT' , 'utf8';

use PDL;
use PDL::Core ':Internal';
use PDL::NiceSlice;

my $hash = {};
   $hash->{0} = "zero";
   $hash->{1} = "one";
   $hash->{2} = "two";

my $pdl = pdl(sort keys %{$hash});

say $pdl;
