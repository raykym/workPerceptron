#!/usr/bin/env perl
#
# dump_structure.txtを読み込んで確認する
#
use strict;
use warnings;
use utf8;
use feature ':5.13';

binmode 'STDOUT' , ':utf8';

use Time::HiRes qw/ time /;
use Data::Dumper;
use Devel::Size;
use Devel::Cycle;

use lib './lib';
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

   # dump_structureを用意したので、読み込みを試す。
    # 試してみたところ、困ったことに、復元は出来ていたが、
    # 学習が不完全で、確率的に成功しているだけということが分かった。。。。
    # 過学習どころか不完全だったことが分かった。。。。

    my $dumpData = require './dump_structure.txt';

    my $multilayer = Multilayer->new();
       my $total_size = Devel::Size::total_size($multilayer);
       Logging("DEBUG: total size: $total_size byte");

       $multilayer->layer_init($dumpData->{layer_init});
        $total_size = Devel::Size::total_size($multilayer);
       Logging("DEBUG: total size: $total_size byte");

       $multilayer->takeover($dumpData);
        $total_size = Devel::Size::total_size($multilayer);
       Logging("DEBUG: total size: $total_size byte");

       # dumpしただけで、learnedになっていないので、
       $multilayer->stat('learned');

       $multilayer->input([ 100 , 100 ]); 
       my  $ret = $multilayer->calc_multi();
       say "!!!! CHECK: @{$ret->[-1]}  hope: 1 -----"; 

       $multilayer->input([ 100 , 0 ]); 
       $ret = $multilayer->calc_multi();
       say "!!!! CHECK: @{$ret->[-1]} hope: 0 -----";  

       $multilayer->input([ 0 , 100 ]); 
       $ret = $multilayer->calc_multi();
       say "!!!! CHECK: @{$ret->[-1]} hope: 0 -----";  

       $multilayer->input([ 0 , 0 ]); 
       $ret = $multilayer->calc_multi();
       say "!!!! CHECK: @{$ret->[-1]} hope: 1 -----";  

       # 単位論理

       $multilayer->input([ 1 , 1 ]); 
       my  $ret = $multilayer->calc_multi();
       say "!!!! CHECK: @{$ret->[-1]}  hope: 1 -----"; 

       $multilayer->input([ 1 , 0 ]); 
       $ret = $multilayer->calc_multi();
       say "!!!! CHECK: @{$ret->[-1]} hope: 0 -----";  

       $multilayer->input([ 0 , 1 ]); 
       $ret = $multilayer->calc_multi();
       say "!!!! CHECK: @{$ret->[-1]} hope: 0 -----";  

       $multilayer->input([ 0 , 0 ]); 
       $ret = $multilayer->calc_multi();
       say "!!!! CHECK: @{$ret->[-1]} hope: 1 -----";  
