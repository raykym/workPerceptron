#!/usr/bin/env perl
#
# simple perceptronの学習ななど、自作実装で動作を確認する。
# AND OR NAND NORについて動作することを確認
#
# 高卒でもわかる機械学習　サイトを参考に作成
#
# Dataplotを使って作図した結果、どうも過学習になっていると推測した
# xorの最小構成と考えていた、layer_member [ 1 , 0 ] 構成で学習率をどんどん下げて、ある程度近いところまで行くことがわかった。　
#
# 他の方法が無いものか探していると、1986年の論文をそのままC言語で実装したサイトを見つけた。
# そこでは順伝送でシグモイド関数を利用していた。
# ReLUの構成では反応が強すぎるのだと、感覚的に思った。
# そこで、構成を２倍にして、出力層も２個のパーセプトロンで論理を分割したところ、収束した
# Multilayer.pmのlimitと学習率をバランス取って、増減させながら、収束できた。
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
		        class => [ 1 , 0 ],
		        input => [ 100 , 100 ]
		      },	
		      {
		        class => [ 0 , 1 ],
			input => [ 0 , 100 ]
		      },
		      {
		        class => [ 0 , 1 ],
			input => [ 100 , 0 ]
		      },
		      {
			class => [ 1 , 0 ],
			input => [ 0 , 0 ]
		      },
	              ];

=pod
=cut

    # ２層パーセプトロンを構成して、XOR回路を学習させる
    # 何回か動かすと何故か失敗することがある。。。？

    my $structure = { 
	              layer_member  => [ 3 , 1 ],
		      input_count => 1 ,
		      learn_rate => 0.00041,
		      layer_act_func => [ 'ReLU' , 'Step' ],
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

=pod
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

       #my $total_size = Devel::Size::total_size($multilayer);
       #Logging("DEBUG: total size: $total_size byte");

