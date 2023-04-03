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

    open ( my $fh , '>' , './onelayer_plotdata.txt');

    #   print $fh Dumper $learndata;

    # x,yを与えて結果をまとめて出力をgnuplotでプロットさせる

    my $point = 0;
    for ( my $x = -10 ; $x <= 10 ; $x+=0.1  ) {
        for ( my $y = -10 ; $y <= 10 ; $y+=0.1  ) {
               $multilayer->input( [ $x , $y ] );
            my $out = $multilayer->calc_multi();
	    #say "onelayer0:  @{$out->[0]} ";
	    #say "onelayer1:  @{$out->[1]} ";
	    #  say "onelayer2:  @{$out->[2]} ";
	    #say " $x $y $out->[1]->[0] ";
	    # 
	    # layer 1の結果を位置をずらして表示する
	    say $fh " $x $y $out->[1]->[$point] ";
	    #$point++;
        }
    }

    close $fh;




