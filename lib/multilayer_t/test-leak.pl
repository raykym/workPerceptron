#!/usr/bin/env perl
#
use strict;
use warnings;
use utf8;
use feature 'say';
binmode 'STDOUT' , ':utf8';

$|=1;

use Devel::Cycle;
use Devel::Size;
use Devel::Peek;

use FindBin;
use lib "$FindBin::Bin/..";
use Perceptron;
use Multilayer;


my $multilayer = Multilayer->new;

my $structure = { layer_member => [ 1 , 1 ],
	          input_count => 3 ,
		  learn_rate => 0.34,
	        };

   $multilayer->layer_init($structure);

my $loop_flg = 1;
my $loop_count = 0;
while ( $loop_flg ) {
    $loop_count++;
    say "loop count: $loop_count";
    if ($loop_count > 2000 ) {
        die "on error";
        $loop_flg = 0;
    }

    my $learndata = [];
    for my $count ( 1 .. 100 ) {
        my $x = int(rand(100));
        my $y = int(rand(100));
        my $z = int(rand(100));
        my $w = int(rand(100));
        my $sample->{class} = [ 1 , 0 ];
	   $sample->{input} = [ $x , $y , $z , $w ];
	push(@{$learndata} , $sample );
    }

   my $stat = $multilayer->learn($learndata);
   say "stat: $stat";

   my $learndata_size_point = Devel::Size::total_size($learndata);
   say "total size point: $learndata_size_point";

   my $total_size_point = Devel::Size::total_size($multilayer);
   say "total size point: $total_size_point";

   say "cycle";
   find_cycle($multilayer);
   say "cycle end";

} # while

