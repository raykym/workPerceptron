#!/usr/bin/env perl
#
# ディープラーニングの表現力を検証する。
# 
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

    # データをきちんと作る必要がある。
    # x,yを入力してzが出力される関数」としてデータを用意する
    # 学習用にデータを抽出して、学習、
    # 結果的にx,yを与えるとｚが返ってくる構造になる
    # sinc(x)を3次元で表現するデータから、学習して、似たものを再現するところまでを試す
    #
    open ( my $fhl , '>' , './onelayer_learndata.txt');

    my $z_array = [];

    for ( my $x = 1 ; $x <= 10 ; $x++  ) {
        for ( my $y = 1 ; $y <= 10 ; $y++  ) {
	    my $z = (sin ( sqrt( $x**2 + $y**2 ) ) /  sqrt( $x**2 + $y**2 )) ;
            push(@{$z_array} , $z);
	    say $fhl "$x $y $z";
        } # for y
    } # for x
    close $fhl;

    # 100個の出力層で個別にクラスを指定する　autoEncoderの状態
    for ( my $x = 1 ; $x <= 10 ; $x++  ) {
        for ( my $y = 1 ; $y <= 10 ; $y++  ) {
            my $sample = {};
               $sample->{input} = [ $x , $y ];
               $sample->{class} = $z_array; 
	       push(@{$learndata} , $sample );
        } # for y
    } # for x

    undef $z_array;

    #say "learndata";
    #print Dumper $learndata;


    my $structure = { 
	              layer_member  => [ 9999 , 99 ],
		      input_count => 1 ,
		      learn_rate => 0.0042,
		      layer_act_func => [ 'Sigmoid' , 'None' , 'None' , 'Sigmoid' , 'None' ],
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


    open ( my $fh , '>' , './onelayer_plotdata.txt');

    #   print $fh Dumper $learndata;

    # x,yを与えて結果をまとめて出力をgnuplotでプロットさせる

    my $point = 0;
    for ( my $x = 1 ; $x <= 10 ; $x++  ) {
        for ( my $y = 1 ; $y <= 10 ; $y++  ) {
               $multilayer->input( [ $x , $y ] );
            my $out = $multilayer->calc_multi();
	    #say "onelayer0:  @{$out->[0]} ";
	    #say "onelayer1:  @{$out->[1]} ";
	    #  say "onelayer2:  @{$out->[2]} ";
	    say $fh " $x $y $out->[-1][$point] ";
	    $point++;
        }
    }

    close $fh;

