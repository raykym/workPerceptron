#!/usr/bin/env perl
#
# 多層パーセプトロンのPDLを利用した版

use strict;
use warnings;
use utf8;
use feature 'say';

binmode 'STDOUT' , ':utf8';

use FindBin;
use lib "$FindBin::Bin/../lib";
use Multilayer_PDL;

use Tie::IxHash;
use Time::HiRes qw / time /;

use PDL;
use PDL::NiceSlice;
use PDL::Core ':Internal';

$|=1;

srand();

sub Logging {
        my $logline = shift;
        my $dt = time();
        say "$dt | $logline";

        undef $dt;
        undef $logline;

        return;
}

sub function_2 {
    my $X = shift;
    $X = topdl($X);

    return $X(0) ** 2 + $X(1) ** 2;
}




my $Multilayer = Multilayer_PDL->new;

my $X = pdl ( 3.0 , 4.0 );

Logging &function_2($X);

Logging $Multilayer->numerical_gradient(\&function_2 , $X );


