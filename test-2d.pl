#!/usr/bin/env perl
#
# simple perceptronの学習ななど、自作実装で動作を確認する。
# AND OR NAND NORについて動作することを確認
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
#use Devel::Peek;

use lib './lib';
use Perceptron;
use Multilayer;

#use Scalar::Util qw/ weaken /;


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

    # y=x+120　でデータが分かれるように、ポイントを指定して、学習を行わせる 

    my $structure = {
                      layer_member  => [ 1 , 1 ],
                      input_count => 3 ,
                      learn_rate => 0.34
                    };

    my $multilayer = Multilayer->new();
       $multilayer->layer_init($structure);


       $multilayer->disp_waits();

    my $multi_learndata_online = undef;

 my $verification_flg = 1;
 my $verification_count = 0;
 my $ver_chk = [];

 while ( $verification_flg ) {
    $verification_count++;

    if ($verification_count >= 20000 ) {
        Logging("DEBUG: verification_count over");
	#die "MAT dump need";
	exit;
        $verification_flg = 0;
    }

       my $learn_flg = 1;  # loop: 1 not loop 0
       my $learn_count = 0;  # サンプルデータとして与えるグループ数、グループ内のデータ数はforループに依存する


       #機械学習 サンプルデータを生成して繰り返す
       while ($learn_flg) { # classラベルが最低1回ずつは学習成功するまで

         # $learn_flg = 0; # 手動処理の場合の追加

           $multi_learndata_online = []; # ループ単位で初期化する

           $learn_count++;
           Logging("DEBUG: learn_count: $learn_count");
           if ($learn_count >= 20000 ) {
               Logging("DEBUG: learn_count over");
	       #die "MAT dump need";
               #exit;
               $learn_flg = 0;
           }

	    srand();
           # データが直線の始点と終点という考えでデータを入力していた。
           # 10個のサンプルを作成して、チェックする学習が完了しない場合はループする
           # 学習が完了しても分類に失敗するケースが多々ある
           for my $count ( 1 .. 100 ) {
               # y = x+120 上のポイント 
               my $x1 = -100 - int(rand(300));
               my $x2 = -100 - int(rand(300));
               my $y1 = int(rand(300));
               my $y2 = int(rand(300));
               my $learndata_a = {};
                  $learndata_a->{class} = [ 1 , 0 ];
                  $learndata_a->{input} = [ $x1 , $y1 , $x2 , $y2 ];
               push(@{$multi_learndata_online} , $learndata_a);

               # y = x+120  下のポイント
                $x1 = int(rand(300));
                $x2 = int(rand(300));
                $y1 = -int(rand(300));
                $y2 = -int(rand(300)) ;
               my  $learndata_b = {};
                  $learndata_b->{class} = [ 0 , 1 ];
                  $learndata_b->{input} = [ $x1 , $y1 , $x2 , $y2 ];
               push(@{$multi_learndata_online} , $learndata_b);

	       undef $learndata_a;
	       undef $learndata_b;
           }

	   my $stat = $multilayer->learn($multi_learndata_online);

           if ($stat eq 'learned') {
               $learn_flg = 0;
               # ここにwhileの下の処理を入れてチェックすればverificationのループは不要だが、思考過程がわかるようにこのままに
              Logging("learn finish!!!");
           } else {
              Logging("learn not yet!");
           }

           undef $multi_learndata_online;
       } # while

       # class毎に成功が出て、学習終了しても、適切に学習出来てないケースが在る
       # そのため、verificationループでラップして、実際に成功するまでループさせることに

       my $x1 = -100 - int(rand(300));
       my $x2 = -100 - int(rand(300));
       my $y1 = int(rand(300));
       my $y2 = int(rand(300));
     #  Logging("DEBUG: x1: $x1 y1: $y1 x2: $x2 y2: $y2");

       $multilayer->input([ $x1 , $y1 , $x2 , $y2 ]);
       my  $ret = $multilayer->calc_multi();
       say "!!!! CHECK: @{$ret->[-1]}  hope: 10 -----";    # 1 ,0 を期待

       my $retstring = join ("" , @{$ret->[-1]});
       push(@{$ver_chk} , 'ok' ) if $retstring eq '10';

       $x1 = int(rand(300));
       $x2 = int(rand(300));
       $y1 = -int(rand(300));
       $y2 = -int(rand(300)) ;
       #   Logging("DEBUG: x1: $x1 y1: $y1 x2: $x2 y2: $y2");

       $multilayer->input([ $x1 , $y1 , $x2 , $y2 ]);
       $ret = $multilayer->calc_multi();
       say "!!!! CHECK: @{$ret->[-1]} hope: 01 -----";    # 0 , 1 を期待

        $retstring = join ("" , @{$ret->[-1]});
       push(@{$ver_chk} , 'ok' ) if $retstring eq '01';

       Logging("DEBUG: verification: $verification_count ");

       # classラベル毎にチェックして"okok"であれば完了
       my $ver_chk_string = join ("" , @{$ver_chk});
       if ( $ver_chk_string eq 'okok' ) {
           # verification ループを止める
           $verification_flg = 0;

           print Dumper $structure;
           $multilayer->disp_waits();
       } else {
           $ver_chk = [];
           $learn_flg = 1; # ループを差し戻す
       }

       undef $x1;
       undef $x2;
       undef $y1;
       undef $y2;
       undef $ret;
       undef $retstring;
       undef $ver_chk_string;

 } # while verification

       $multilayer->dump_structure();


