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
#use EV;
#use AnyEvent;
use List::Util;

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

    #パラメータ設定
    my $structure = { 
	    #  layer_member  => [ 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 ],
	              layer_member  => [ 2 , 1 , 0 ],
		      input_count => 1 ,
		      learn_rate => 0.01,
           #  layer_act_func => [ 'Sigmoid' , 'Sigmoid' , 'Sigmoid' , 'Sigmoid' , 'Sigmoid' , 'Sigmoid' , 'Sigmoid' , 'Sigmoid' , 'Sigmoid' , 'None' ],
	   #  layer_act_func => [ 'ReLU' , 'ReLU' , 'ReLU' , 'ReLU' , 'ReLU' , 'ReLU' , 'ReLU' , 'ReLU' , 'ReLU' , 'None' ],
	   #layer_act_func => [ 'Sigmoid' , 'None' ],
	              layer_act_func => [ 'Sigmoid' , 'Sigmoid' , 'None' ],
		      optimaizer => 'adam' ,
	            };

    my $picup_cnt = 20000; # ピックアップデータ数
    my $batch = 50;   # バッチ数
    my $intre = int( $picup_cnt / $batch );  # インテレート数
    my $epoc = 250;




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

    # learndataの抽出
    open ( my $fhl , '>' , './onelayer_learndata.txt');

    my @tmp = @{$createdata};
    my $data_cnt = $#tmp;
    undef @tmp;

#    my $picup_cnt = 40000; # ピックアップデータ数

    for my $cnt ( 1 .. $picup_cnt ) {
        my $choice = int(rand($data_cnt));
	my $sample = $createdata->[$choice];
        push(@{$learndata} , $sample);
	 #say $fhl "$x $y $z";
	say $fhl "$sample->{input}->[0] $sample->{input}->[1] @{$sample->{class}}";
    } #for cnt

    close $fhl;

    undef $createdata; # メモリ開放

#    my $batch = 500;   # バッチ数
#    my $intre = int( $picup_cnt / $batch );  # インテレート数

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

    my $multilayer = Multilayer->new();
       $multilayer->layer_init($structure);
     
    my $ts1 = total_size($multilayer);  
       Logging("1. multilayer  total_size: $ts1 ");

       #   $multilayer->disp_waits();


    $interater = &input_layer($interater , $intre , $batch );

=pod # input_layerデバッグ用
        for my $batch (@{$interater}) {
            for my $sample (@{$batch}) {
                say @{$sample->{input}};
                say @{$sample->{class}};
	    }
        }

    exit;
=cut

    # 正規化したデータを確認する
    open ( my $fh2 , '>' , './onelayer_inter.txt');
    for my $batch (@{$interater}) {
        for my $sample (@{$batch}) {
            say $fh2 "$sample->{input}->[0] $sample->{input}->[1] $sample->{class}->[0]";
        }
    }
    close $fh2;

sub input_layer {
        my ($interater , $intre , $batch ) = @_;
    #入力層を標準化する　0-1にまとめる
    # {input}と{class}を変更する

        my @list_input = (); #全てのinput
        my @list_class = (); #全てのclass
        for my $batch (@{$interater}) {
            for my $sample (@{$batch}) {
                push(@list_input , @{$sample->{input}});
                push(@list_class , @{$sample->{class}});
	    }
        }
        my $min_input = List::Util::min(@list_input);
        my $max_input = List::Util::max(@list_input);
        my $min_class = List::Util::min(@list_class);
        my $max_class = List::Util::max(@list_class);

	undef @list_input;
	undef @list_class;

	my $input_offset = undef;
	my $input_width = undef;

	if ($min_input < 0 ) {
           $input_offset = abs($min_input);
           $input_width = $max_input + $input_offset;
        } else {
           $input_offset = $min_input; 
           $input_width = $max_input - $min_input;
        }

	my $class_offset = undef;
	my $class_width = undef;

	if ($min_class < 0 ) {
            $class_offset = abs($min_class);
            $class_width = $max_class + $class_offset;
	} else {
            $class_offset = $min_class;
	    $class_width = $max_class - $min_class;
	}

        #標準化
        for (my $i=0; $i <= $intre - 1; $i++) {
	    for (my $j=0 ; $j <= $batch - 1; $j++) {
		    #  @{$interater->[$i]->[$j]->{input}} = map { ($_ + $input_offset ) / $input_width } @{$interater->[$i]->[$j]->{input}};
		    #  @{$interater->[$i]->[$j]->{class}} = map { ($_ + $class_offset ) / $class_width } @{$interater->[$i]->[$j]->{class}};
		@{$interater->[$i]->[$j]->{input}} = map { ($_ - $min_input ) / ($max_input - $min_input )} @{$interater->[$i]->[$j]->{input}};
	        @{$interater->[$i]->[$j]->{class}} = map { ($_ - $min_class ) / ($max_class - $min_class )} @{$interater->[$i]->[$j]->{class}};
            }
        }

        return $interater;
    } # input_layer


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

    #    $multilayer->datalog_transaction('on'); #datalogをトランザクションモードで高速化する
    
#    my $epoc = 500;

      $multilayer->datalog_init();
      $multilayer->datalog_snapshot(); # 学習前状態


    for my $epoc_cnt ( 1 .. $epoc ) {  
	my $loss = undef;
        # バッチ毎に学習 
        for (my $idx = 0 ; $idx <= $intre -1 ; $idx++){

           $multilayer->learn($interater->[$idx]);

	   $loss = $multilayer->loss($interater->[$idx]); # sample->{class}が必要　
	   Logging(" epoc: $epoc_cnt batch: $idx 誤差関数 $loss ");


           if ( $loss <= 1e-20 ) {
	       # 誤差関数が一定以上に下がったら
               last;
           }
        }

        # epoc毎にsnapshotを取得する
	    $multilayer->datalog_snapshot();

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

