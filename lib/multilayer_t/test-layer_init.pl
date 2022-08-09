#!/usr/bin/env perl
#
#
#
use strict;
use warnings;
use utf8;
use feature 'say';

binmode 'STDOUT' , ':utf8';

use FindBin;
use lib "$FindBin::Bin/..";

use Multilayer;


my $obj = Multilayer->new;

my $structure = {
                    layer_member => [ 1 , 0 ],
		    input_count => 1,
                    learn_rate => 0.004 , 
                    layer_act_func => [ 'ReLU' , 'Step' ],
               };

   $obj->layer_init($structure);

   $obj->disp_waits();

   $obj->stat('learned');

   $obj->input( [1000 , 1000] );

   my $res = $obj->calc_multi();

   say @{$_} for @{$res};
