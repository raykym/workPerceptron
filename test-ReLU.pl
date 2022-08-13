#!/usr/bin/env perl
#
# ReLU関数の多層パーセプトロンで表現力というものを確認する
# 分類のテスト
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

# ２群のデータを入れて、分類を試す

    my $point1 = [ 0 , 10 ];
    my $point2 = [ 20 , 0 ];
    my $group1 = [];  # class [ 1 , 0 ]
    my $group2 = [];  # class [ 0 , 1 ]

    srand();

    for my $cnt ( 1 .. 100 ) {
        my $sample->{class} = [ 1 , 0 ];
           $sample->{input} = [ $point1->[0] + rand(10) , $point1->[1] + rand(10) ] ;
        push (@{$group1} , $sample);
    }
    for my $cnt ( 1 .. 100 ) {
         my $sample->{class} = [ 0 , 1 ];
            $sample->{input} = [ $point2->[0] + rand(10) , $point2->[1] + rand(10) ] ;
        push (@{$group2} , $sample );
    }

    # 教育用抽出
    my $learndata = [];
    while ( 1 ) {

        my @tmp1 = @{$group1};
	my @tmp2 = @{$group2};

        my $index_count1 = $#tmp1;
        my $index_count2 = $#tmp2;

	if ($#tmp1 <= 49 ) {
            last;
	}

	my $index1 = int(rand($index_count1));
	my $index2 = int(rand($index_count2));

        my @tmp = splice(@{$group1} , $index1 , 1 );
	push (@{$learndata} , @tmp );

         @tmp = splice(@{$group2} , $index2 , 1 );
	push (@{$learndata} , @tmp );
    }

    # plot data output
    open ( my $fh , '>' , 'learndata_plot.txt' );
    say $fh "# plot learn data";
    for my $sample ( @{$learndata} ) {
        say $fh "$sample->{input}->[0] $sample->{input}->[1] "; 
    }
    close($fh);

    my $structure = { 
	              layer_member  => [  1 , 1 ],
		      input_count => 1 ,
		      learn_rate => 0.00041,
		      layer_act_func => [ 'ReLU' , 'Step' ],
	            };


    my $multilayer = Multilayer->new();
       $multilayer->layer_init($structure);

       $multilayer->disp_waits();

       $multilayer->datalog_transaction('on'); #datalogをトランザクションモードで高速化する

       $multilayer->learn($learndata);

       $multilayer->disp_waits();

       # 学習結果を確認する
       for my $sample ( @{$learndata}) {
           $multilayer->stat('learned'); # statを強制変更	       
	   $multilayer->input($sample->{input});    
           my $ret = $multilayer->calc_multi();
           say "out: @{$ret->[-1]}  class: @{$sample->{class}} ";
       }	       

       # 同時作成したデータで予測を実施

       my @checkdata = (@{$group1} , @{$group2} );

       for my $sample (@checkdata) {
           $multilayer->stat('learned'); # statを強制変更	       
	   $multilayer->input($sample->{input});    
           my $ret = $multilayer->calc_multi();
           say "out: @{$ret->[-1]}  class: @{$sample->{class}} ";
       }	       

       # plot data output
       open ( $fh , '>' , 'checkdata_plot.txt' );
       say $fh "# plot check data";
       for my $sample ( @{$learndata} ) {
	   say $fh "$sample->{input}->[0] $sample->{input}->[1] "; 
       }
       close($fh);


       $multilayer->dump_structure();


