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
use MultiLayerNet;
use Adam_optimizer;

my $hparams = {};
   # 引数のファイル名を展開する
   $hparams = retrieve($ARGV[0]);

   for my $key (keys %{$hparams} ) {

       if ( $hparams->{$key} =~ /^[0-9]+$|^[0-9]+\.[0-9]+$/ ) {
           say "$key : $hparams->{$key}"; 
       } else {
           # 数値で無ければスルー
           next;
       }

   }

   open ( my $fh , '>' , './sinc_plotdata.txt');
   for ( my $x = -20 ; $x <= 20 ; $x++  ) {
       for ( my $y = -20 ; $y <= 20 ; $y++  ) {
           my $RET = $hparams->{network}->predict(pdl([ $x , $y ]));
           #say "(loss)RET: $RET";
           my @out = list($RET);
           say $fh " $x $y $out[0] ";
       }
   }
   close($fh);




