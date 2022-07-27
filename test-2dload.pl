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
use Perceptron;
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

       my $x1 = 100 - int(rand(300));
       my $x2 = 100 - int(rand(300));
       my $y1 = int(rand(300));
       my $y2 = int(rand(300)); 
       Logging("DEBUG: x1: $x1 y1: $y1 x2: $x2 y2: $y2");

       $multilayer->input([ $x1 , $y1 , $x2 , $y2 ]); 
       my  $ret = $multilayer->calc_multi();
       say "!!!! CHECK: @{$ret->[-1]}  hope: 10 -----";    # 1 ,0 を期待

        $x1 = int(rand(300));
        $x2 = int(rand(300));
        $y1 = -int(rand(300));
        $y2 = -int(rand(300));
       Logging("DEBUG: x1: $x1 y1: $y1 x2: $x2 y2: $y2");

       $multilayer->input([ $x1 , $y1 , $x2 , $y2 ]); 
       $ret = $multilayer->calc_multi();
       say "!!!! CHECK: @{$ret->[-1]} hope: 01 -----";    # 0 , 1 を期




