#!/usr/bin/env perl
#
# ReLU関数の多層パーセプトロンで表現力というものを確認する
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





    my $structure = { 
	              layer_member  => [  0 , 0 ],
		      input_count => 1 ,
		      learn_rate => 0.00041,
		      layer_act_func => [ 'ReLU' , 'ReLU' ],
	            };


    my $multilayer = Multilayer->new();
       $multilayer->layer_init($structure);

       $multilayer->disp_waits();

       $multilayer->datalog_transaction('on'); #datalogをトランザクションモードで高速化する

       $multilayer->learn($multi_learndata_test);

       $multilayer->disp_waits();

       # 学習結果を確認する
       for my $sample ( @{$multi_learndata_test}) {
           $multilayer->stat('learned'); # statを強制変更	       
	   $multilayer->input($sample->{input});    
           my $ret = $multilayer->calc_multi();
           say "out: @{$ret->[-1]}  class: @{$sample->{class}} ";
       }	       

       # 同時作成したデータで予測を実施
       for my $sample ( @{$multi_checkdata_test}) {
           $multilayer->stat('learned'); # statを強制変更	       
	   $multilayer->input($sample->{input});    
           my $ret = $multilayer->calc_multi();
           say "out: @{$ret->[-1]}  class: @{$sample->{class}} ";
       }	       



       $multilayer->dump_structure();


