#!/usr/bin/env perl
#
# AutoEncoderを試する  ->うまくいかない
# 
#
use strict;
use warnings;
use utf8;
use feature ':5.13';

binmode 'STDOUT' , ':utf8';

use Time::HiRes qw/ time /;
use Data::Dumper;
#use Devel::Size;
#use Devel::Cycle;

use FindBin;
use lib "$FindBin::Bin/lib";

#use Perceptron;
use Multilayer;


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

# 学習データ
    my $learndata = []; 

    my $sample = {};
       $sample->{class} = [ 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 ];	    
       $sample->{input} = [ 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 ];

       push(@$learndata , $sample);

    my $structure = { 
	              layer_member  => [  8 , 3 , 8 ],
		      input_count => 8 ,
		      learn_rate => 0.0041,
		      layer_act_func => [ 'ReLU' , 'ReLU' , 'ReLU' ],
	            };


    my $multilayer = Multilayer->new();
       $multilayer->layer_init($structure);

       #   $multilayer->disp_waits();

       $multilayer->datalog_transaction('on'); #datalogをトランザクションモードで高速化する

       $multilayer->learn($learndata);

       #  $multilayer->disp_waits();
       
       $multilayer->stat('learned'); #強制モード変更

       $multilayer->input($sample->{input});
    my $out = $multilayer->calc_multi();

       say "out: @{$out->[-1]} class: @{$sample->{class}}";

       $multilayer->dump_structure();


