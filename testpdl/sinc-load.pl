#!/usr/bin/env perl
#
# twolayernet.hparamsを読み込んで使う
#

use v5.32;
use utf8;

binmode 'STDOUT' , ':utf8';

use PDL;
use PDL::Core ':Internal';
use PDL::NiceSlice;

use Storable qw / store retrieve /;
use PDL::IO::Storable;

use FindBin;
use lib "$FindBin::Bin::../lib";
use TwoLayerNet;
use Adam_optimizer;

my $hparams = {};

   $hparams = retrieve('twolayernet.hparams');

   for my $key (keys %{$hparams} ) {

       if ( $hparams->{$key} =~ /^[0-9]+$|^[0-9]+\.[0-9]+$/ ) {
           say "$key : $hparams->{$key}"; 
       } else {
           # 数値で無ければスルー
           next;
       }

   }

   open ( my $fh , '>' , './sinc_plotdata.txt');
   for ( my $x = -10 ; $x <= 10 ; $x++  ) {
       for ( my $y = -10 ; $y <= 10 ; $y++  ) {
           my $RET = $hparams->{network}->predict(pdl([ $x , $y ]));
           #say "(loss)RET: $RET";
           my @out = list($RET);
           say $fh " $x $y $out[0] ";
       }
   }
   close($fh);




