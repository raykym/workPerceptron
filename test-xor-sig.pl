#!/usr/bin/env perl
#
# ReLUとStepの構成でバックプロパゲーション処理を作ったが、
# 単純なXOR回路を試すためにSigmoid関数を入れてみたが、これは相性が悪いらしい。
# 別の処理系を用意しないと収束できない気がする。
#
use strict;
use warnings;
use utf8;
use feature ':5.13';

binmode 'STDOUT' , ':utf8';

use Time::HiRes qw/ time /;
#use Data::Dumper;
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



		    # multilayer用 clasはs0,1 に変更される。
		    # あえて大きな数値を入れて大体の感じでXORを表現する
		    # 2層構造のXORは最低100を入力しないと学習出来ない
		    #
     my $multi_learndata_XORgate = [
	              { 
		        class => [ 1 ],
		        input => [ 1 , 1 ]
		      },	
		      {
		        class => [ 0 ],
			input => [ 0 , 1 ]
		      },
		      {
		        class => [ 0 ],
			input => [ 1 , 0 ]
		      },
		      {
			class => [ 1 ],
			input => [ 0 , 0 ]
		      },
	              ];

=pod
=cut

    # ２層パーセプトロンを構成して、XOR回路を学習させる
    # 何回か動かすと何故か失敗することがある。。。？

    my $structure = { 
	              layer_member  => [ 2 , 0 ],
		      input_count => 1 ,
		      learn_rate => 0.0009,
		      layer_act_func => [ 'Sigmoid' , 'Sigmoid' ],
	            };


    my $multilayer = Multilayer->new();
       $multilayer->layer_init($structure);

       $multilayer->disp_waits();

       $multilayer->datalog_transaction('on'); #datalogをトランザクションモードで高速化する

       $multilayer->learn($multi_learndata_XORgate);

       $multilayer->disp_waits();

       # 学習結果を確認する
       for my $sample ( @{$multi_learndata_XORgate}) {
           $multilayer->stat('learned'); # statを強制変更	       
	   $multilayer->input($sample->{input});    
           my $ret = $multilayer->calc_multi();
           say "out: @{$ret->[-1]}  class: @{$sample->{class}} ";
       }	       

       $multilayer->dump_structure();


