#!/usr/bin/env perl
#
# 表現力がわからないのでテストする
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
    my $createdata = []; # 作成全データ
    my $learndata = [];  # 10000個ピックアップ
    my $interater = []; # ２次元配列  バッチ500毎に分割 500x20 (499x19)



    my $structure = { 
	    #  layer_member  => [ 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 0 ],
	    #  layer_member  => [ 499 , 0 ],
	    #  layer_member  => [ 2 , 2 , 0 ],
	              layer_member  => [ 0 , 0 , 0 ],
		      input_count => 1 ,
		      learn_rate => 0.000001,
           #  layer_act_func => [ 'Sigmoid' , 'Sigmoid' , 'Sigmoid' , 'Sigmoid' , 'Sigmoid' , 'Sigmoid' , 'Sigmoid' , 'Sigmoid' , 'Sigmoid' , 'None' ],
	   #  layer_act_func => [ 'ReLU' , 'ReLU' , 'ReLU' , 'ReLU' , 'ReLU' , 'ReLU' , 'ReLU' , 'ReLU' , 'ReLU' , 'None' ],
	   #  layer_act_func => [ 'ReLU' , 'ReLU' , 'None' ],
	   #  layer_act_func => [ 'Sigmoid' , 'Sigmoid' , 'None' ],
	   #  layer_act_func => [ 'Sigmoid' ],
	              layer_act_func => [ 'None' , 'Sigmoid' , 'None' ],
	   #  layer_act_func => [ 'ReLU' , 'ReLU' , 'None' ],
		      optimaizer => 'adam' ,
	            };


    my $multilayer = Multilayer->new();
       $multilayer->layer_init($structure);

       $multilayer->disp_waits();

    open ( my $fh , '>' , './plot-plotdata.txt');

    #   print $fh Dumper $learndata;

    # x,yを与えて結果をまとめて出力をgnuplotでプロットさせる

    for ( my $x = -10 ; $x <= 10 ; $x++  ) {
        for ( my $y = -10 ; $y <= 10 ; $y++  ) {
               $multilayer->input( [ $x , $y ] );
            my $out = $multilayer->calc_multi('learn');
	    say $fh " $x $y $out->[-1]->[0] ";
        }
    }

    close $fh;

