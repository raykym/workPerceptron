#!/usr/bin/env perl
#
#Sincpdlのテスト
#

use v5.32;
use utf8;

binmode 'STDOUT' , ';utf8';

$|=1;

use FindBin;
use lib "$FindBin::Bin/lib";

use Sincpdl;

use PDL;
use PDL::Core ':Internal';
use PDL::NiceSlice;

my $makedata = Sincpdl->new;
my ($train_x , $train_t) = $makedata->make;

say $train_x->dims;
say $train_t->dims;
