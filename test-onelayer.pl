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
    my $createdata = []; # 作成全データ
    my $learndata = [];  # 10000個ピックアップ
    my $interater = []; # ２次元配列  バッチ500毎に分割 500x20 (499x19)

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
           }
            my $sample = {};
               $sample->{input} = [ $x , $y ];
               $sample->{class} = $z; 
	       push(@{$createdata} , $sample );
        } # for y
    } # for x

    # learndataの抽出
    open ( my $fhl , '>' , './onelayer_learndata.txt');

    my @tmp = @{$createdata};
    my $data_cnt = $#tmp;
    undef @tmp;

    my $picup_cnt = 10000; # ピックアップデータ数

    for my $cnt ( 1 .. $picup_cnt ) {
        my $choice = int(rand($data_cnt));
	my $sample = $createdata->[$choice];
        push(@{$learndata} , $sample);
	 #say $fhl "$x $y $z";
	say $fhl "$sample->{input}->[0] $sample->{input}->[1] $sample->{class}";
    } #for cnt

    close $fhl;

    undef $createdata; # メモリ開放

    my $batch = 500;   # バッチ数
    my $intre = int( 10000 / $batch );  # インテレート数

    #バッチに分割
    for my $i ( 1 .. $intre ) {
	my $tmp = [];
        for my $j ( 1 .. $batch ) {
	    push(@{$tmp} , shift(@{$learndata}));
        }
        push(@{$interater} , $tmp);
	undef $tmp;
    }

    undef $learndata;

    #say "learndata";
    #print Dumper $learndata;


    my $structure = { 
	    #  layer_member  => [ 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 0 ],
	              layer_member  => [ 9 , 0 ],
	    #  layer_member  => [ 9 , 9 , 0 ],
	    #  layer_member  => [ 0 ],
		      input_count => 1 ,
		      learn_rate => 0.001,
           #  layer_act_func => [ 'Sigmoid' , 'Sigmoid' , 'Sigmoid' , 'Sigmoid' , 'Sigmoid' , 'Sigmoid' , 'Sigmoid' , 'Sigmoid' , 'Sigmoid' , 'None' ],
	   #  layer_act_func => [ 'ReLU' , 'ReLU' , 'ReLU' , 'ReLU' , 'ReLU' , 'ReLU' , 'ReLU' , 'ReLU' , 'ReLU' , 'None' ],
	   #  layer_act_func => [ 'ReLU' , 'ReLU' , 'None' ],
	   #  layer_act_func => [ 'Sigmoid' , 'Sigmoid' , 'None' ],
	   #  layer_act_func => [ 'None' ],
	              layer_act_func => [ 'Sigmoid' , 'Sigmoid' ],
	   #  layer_act_func => [ 'ReLU' , 'ReLU' , 'None' ],
		      optimaizer => 'adam' ,
	            };


    my $multilayer = Multilayer->new();
       $multilayer->layer_init($structure);
     
    my $ts1 = total_size($multilayer);  
       Logging("1. multilayer  total_size: $ts1 ");

       #   $multilayer->disp_waits();

    # 学習前に入力して出力を確認する
    open ( my $fh1 , '>' , './onelayer_nolearn.txt');
    for ( my $x = -10 ; $x <= 10 ; $x++  ) {
        for ( my $y = -10 ; $y <= 10 ; $y++  ) {
               $multilayer->input( [ $x , $y ] );
            my $out = $multilayer->calc_multi('learn');
	    say $fh1 " $x $y $out->[-1]->[0] ";
        }
    }
    close $fh1;

    $multilayer->datalog_transaction('on'); #datalogをトランザクションモードで高速化する
    my $epoc = 10;

    for my $epoc ( 1 .. $epoc ) {  
	my $loss = undef;
        # バッチ毎に学習 バッチサイズ500 イテレーション数20
        for (my $idx = 0 ; $idx <= 19 ; $idx++){
           $multilayer->learn($interater->[$idx]);

	   $loss = $multilayer->loss();
	   Logging(" epoc: $epoc batch: $idx 誤差関数 $loss ");

           if ( $loss <= 1e-04 ) {
               last;
           }
        }
        if ( $loss <= 1e-04 ) {
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

    #   print $fh Dumper $learndata;

    # x,yを与えて結果をまとめて出力をgnuplotでプロットさせる

    my $po = 0;
    for ( my $x = -10 ; $x <= 10 ; $x++  ) {
        for ( my $y = -10 ; $y <= 10 ; $y++  ) {
               $multilayer->input( [ $x , $y ] );
            my $out = $multilayer->calc_multi();
	    #say $fh " $x $y $out->[0][$po] ";
	    #say "onelayer1:  @{$out->[1]} ";
	    #  say "onelayer2:  @{$out->[2]} ";
	    say $fh " $x $y $out->[-1]->[0] ";

	    #$po++;
        }
    }

    close $fh;

