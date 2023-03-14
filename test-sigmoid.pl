#!/usr/bin/env perl
#
# ディープラーニングの表現力を検証する。
# sigmoid 
#
use strict;
use warnings;
use utf8;
use feature ':5.13';

binmode 'STDOUT' , ':utf8';

use Time::HiRes qw/ time /;
use Data::Dumper;
use Devel::Size qw/ size total_size /;
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

    #
    open ( my $fhl , '>' , './onelayer_sigmoid.txt');

    for ( my $x = 1 ; $x <= 10 ; $x++  ) {
        my $sample = {};
           $sample->{input} = [ $x ];
	   my $z = ( sin ($x) )  ;
	   $sample->{class} = [ $z ];
        push(@{$learndata} , $sample);
	say $fhl " $x $z ";
    } # for x

    close $fhl;

    #say "learndata";
    #print Dumper $learndata;


    my $structure = { 
	              layer_member  => [ 9 , 0 ],
		      input_count => 1 ,
		      learn_rate => 0.0042,
		      layer_act_func => [ 'Sigmoid' , 'Sigmoid' , 'Sigmoid' ],
	            };


    my $multilayer = Multilayer->new();
       $multilayer->layer_init($structure);
     
    my $ts1 = total_size($multilayer);  
       Logging("1. multilayer  total_size: $ts1 ");

       #   $multilayer->disp_waits();

       $multilayer->datalog_transaction('on'); #datalogをトランザクションモードで高速化する

       $multilayer->learn($learndata);

       #  $multilayer->disp_waits();
       $multilayer->dump_structure();

    my $ts2 = total_size($multilayer);  
       Logging("2. multilayer  total_size: $ts2");
    my $diff_ts = $ts2 - $ts1;
       Logging("multilayer gain: $diff_ts");
       
       $multilayer->stat('learned'); #強制モード変更


    open ( my $fh , '>' , './sigmoid_plotdata.txt');

    #   print $fh Dumper $learndata;

    # x,yを与えて結果をまとめて出力をgnuplotでプロットさせる
    for ( my $x = 1 ; $x <= 10 ; $x++  ) {
               $multilayer->input( [ $x ] );
            my $out = $multilayer->calc_multi();
	    #say "onelayer0:  @{$out->[0]} ";
	    #say "onelayer1:  @{$out->[1]} ";
	    #  say "onelayer2:  @{$out->[2]} ";
	    say $fh " $x @{$out->[-1]} ";
    }

    close $fh;

