#!/usr/bin/env perl
#
# ディープラーニングの表現力を検証する。
# 
# testwork用
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
#use EV;
#use AnyEvent;
#use List::Util;

use FindBin;
use lib "$FindBin::Bin/../lib";

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

    #パラメータ設定
    my $structure = { 
	              layer_member  => [ 3 , 3 , 0 ],
		      input_count => 1 ,
		      learn_rate => 0.005,
	   # layer_act_func => [ 'Sigmoid' , 'None' ],
	              layer_act_func => [ 'Sigmoid' , 'Sigmoid' , 'None' ],
		      optimaizer => 'adam' ,
		      picup_cnt => 10000,
		      batch => 25,
		      itre => undef ,
		      epoc => 500,
	            };

    my $picup_cnt = $structure->{picup_cnt}; # ピックアップデータ数
    my $batch = $structure->{batch};   # バッチ数
    my $itre = int( $picup_cnt / $batch );  # イテレート数
    my $epoc = $structure->{epoc};

    my $multilayer = Multilayer->new();
       $multilayer->layer_init($structure);
     
    my $ts1 = total_size($multilayer);  
       Logging("1. multilayer  total_size: $ts1 ");



# 学習データ
    my $createdata = []; # 作成全データ
    my $iterater = []; # ２次元配列  バッチ毎に分割 

    # データをきちんと作る必要がある。
    # x,yを入力してzが出力される関数」としてデータを用意する
    # 学習用にデータを抽出して、学習、
    # 結果的にx,yを与えるとｚが返ってくる構造になる
    # sinc(x)を3次元で表現するデータから、学習して、似たものを再現するところまでを試す
    #

    for ( my $x = -10 ; $x <= 10 ; $x+=0.1  ) {
        for ( my $y = -10 ; $y <= 10 ; $y+=0.1  ) {
	   my $z = undef;
	   if ( $x == 0 && $y == 0 ) {
               $z = 0;
	   } else {
	       $z = (sin ( sqrt( $x**2 + $y**2 ) ) /  sqrt( $x**2 + $y**2 ));
	       #$z = sin($x ** 2+ $y ** 2);
	       #$z = $x ** 2+ $y ** 2;
           }
            my $sample = {};
               $sample->{input} = [ $x , $y ];
	       $sample->{class} = [ $z ];       # 単一でもARRAYrefにする必要がある
	       #$sample->{class} = [ $z , $z , $z , $z , $z ]; 
=pod
               $sample->{class} = []; 
               for my $cnt ( 0 .. 1 ) {
                   push(@{$sample->{class}} , $z);
	       }
=cut

	       push(@{$createdata} , $sample );
        } # for y
    } # for x

    $multilayer->all_learndata($createdata);

    undef $createdata; # メモリ開放

    #  $multilayer->datalog_init();
    #  $multilayer->datalog_snapshot(); # 学習前状態


    for my $epoc_cnt ( 1 .. $epoc ) {  
	my $loss = undef;

	# epocの度にデータをサンプリングし直す
        $multilayer->prep_learndata();
        # 標準化する ReLUでは使ったほうが変化が出る
	#$iterater = $multilayer->input_layer();
	#バッチ正規化はウェイトの初期化と相克らしい 
	$iterater = $multilayer->get_iterater();

        # バッチ毎に学習 
        for (my $idx = 0 ; $idx <= $itre -1 ; $idx++){

           $multilayer->learn($iterater->[$idx]);

	   $loss = $multilayer->loss($iterater->[$idx]); # sample->{class}が必要　
	   Logging(" epoc: $epoc_cnt batch: $idx 誤差関数 $loss ");


           if ( $loss <= 1e-20 ) {
	       # 誤差関数が一定以上に下がったら
               last;
           }
        }

        # epoc毎にsnapshotを取得する
	#    $multilayer->datalog_snapshot();

        if ( $loss <= 1e-20 ) {
	   Logging(" Interupt... " );
           last;
        }
    }



       #  $multilayer->disp_waits();
       $multilayer->dump_structure();

    my $ts2 = total_size($multilayer);  
       Logging("2. multilayer  total_size: $ts2");
    my $diff_ts = $ts2 - $ts1;
       Logging("multilayer gain: $diff_ts");
       
       $multilayer->stat('learned'); #強制モード変更


    open ( my $fh , '>' , './onelayer_plotdata.txt');
    # x,yを与えて結果をまとめて出力をgnuplotでプロットさせる

    for ( my $x = -10 ; $x <= 10 ; $x++  ) {
        for ( my $y = -10 ; $y <= 10 ; $y++  ) {
               $multilayer->input( [ $x , $y ] );
            my $out = $multilayer->calc_multi();
	    say $fh " $x $y $out->[-1]->[0] ";
        }
    }

    close $fh;

