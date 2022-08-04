#!/usr/bin/env perl
#
# simple perceptronの学習ななど、自作実装で動作を確認する。
# AND OR NAND NORについて動作することを確認
#
# 高卒でもわかる機械学習　サイトを参考に作成
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

use FindBin;
use lib "$FindBin::Bin/lib";

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



		    # multilayer用 clasはs0,1 に変更される。
		    # あえて大きな数値を入れて大体の感じでXORを表現する
		    # 2層構造のXORは最低100を入力しないと学習出来ない
		    #
     my $multi_learndata_XORgate = [
	              { 
		        class => [ 1 ],
		        input => [ 100 , 100 ]
		      },	
		      {
		        class => [ 0 ],
			input => [ 0 , 100 ]
		      },
		      {
		        class => [ 0 ],
			input => [ 100 , 0 ]
		      },
		      {
			class => [ 1 ],
			input => [ 0 , 0 ]
		      },
	              ];

    # ２層パーセプトロンを構成して、XOR回路を学習させる
    # 何回か動かすと何故か失敗することがある。。。？

    my $structure = { 
	              layer_member  => [ 0 , 0 , 0 , 0 , 0 ],
		      input_count => 1 ,
		      learn_rate => 0.34
	            };

   my $datalog = Datalog->new();



    my $multilayer = Multilayer->new();
       $multilayer->layer_init($structure);

       $multilayer->disp_waits();

       $multilayer->datalog_transaction('on'); #datalogをトランザクションモードで高速化する

       $multilayer->learn($multi_learndata_XORgate);

       $multilayer->disp_waits();

       # 大きな数値を入れるとXORの動作をしている

       say "input [ 1000 , 1000 ]";
       $multilayer->input([1000 , 1000 ]);
       my $ret = $multilayer->calc_multi();
       say @{$ret->[-1]};

       say "input [ 1000 , 0 ]";
       $multilayer->input([1000 , 0 ]);
        $ret = $multilayer->calc_multi();
       say @{$ret->[-1]};

       say "input [ 0 , 1000 ]";
       $multilayer->input([1000 , 0 ]);
        $ret = $multilayer->calc_multi();
       say @{$ret->[-1]};

       # 単位マトリクスではXORの動作をしない

       say "input [ 0 , 0 ]";
       $multilayer->input([0 , 0 ]);
        $ret = $multilayer->calc_multi();
       say @{$ret->[-1]};

=pod

       say "input [ 1 , 0 ]";
       $multilayer->input([1 , 0 ]);
        $ret = $multilayer->calc_multi();
       say @{$ret->[-1]};

       say "input [ 0 , 1 ]";
       $multilayer->input([0 , 1 ]);
        $ret = $multilayer->calc_multi();
       say @{$ret->[-1]};

       say "input [ 1 , 1 ]";
       $multilayer->input([1 , 1 ]);
        $ret = $multilayer->calc_multi();
       say @{$ret->[-1]};
=cut

       my $total_size = Devel::Size::total_size($multilayer);

       Logging("DEBUG: total size: $total_size byte");

