#!/usr/bin/env perl
#
#  3次元モデルに挑戦
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
#use Scalar::Util qw/ weaken /;

use lib './lib';
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

    #　X , y , zでデータが分かれるように、ポイントを指定して、学習を行わせる 

    my $structure = {
                      layer_member  => [ 9 , 9 , 1 ],
                      input_count => 2 ,
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
           if ($learn_count >= 2000 ) {
               Logging("DEBUG: learn_count over");
	       #die "MAT dump need";
               exit;
               $learn_flg = 0;
           }

	    srand();

           for my $count ( 1 .. 100 ) {

               my $x1 = 1000 - int(rand(2000));
               my $y1 = 1000 - int(rand(2000));
	       my $d1 = int(rand(500));
               my $z1 = sin($x1) * cos($y1) + $d1; 
               my $learndata_a = {};
                  $learndata_a->{class} = [ 1 , 0 ];
                  $learndata_a->{input} = [ $x1 , $y1 , $z1 ];
               push(@{$multi_learndata_online} , $learndata_a);

               my $x2 = 1000 - int(rand(2000));
               my $y2 = 1000 - int(rand(2000));
	       my $d2 = int(rand(500));
               my $z2 = sin($x2) * cos($y2) - $d2 ;
               my  $learndata_b = {};
                  $learndata_b->{class} = [ 0 , 1 ];
                  $learndata_b->{input} = [ $x2 , $y2 , $z2 ];
               push(@{$multi_learndata_online} , $learndata_b);

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

       my $x1 = 1000 - int(rand(2000));
       my $y1 = 1000 - int(rand(2000));
       my $d1 = int(rand(500));
       my $z1 = sin($x1) * cos($y1) + $d1;

       $multilayer->input([ $x1 , $y1 , $z1 ]);
       my  $ret = $multilayer->calc_multi();
       say "!!!! CHECK: @{$ret->[-1]} hope: 10 -----"; 

       my $retstring = join ("" , @{$ret->[-1]});
       push(@{$ver_chk} , 'ok' ) if $retstring eq '10';

       my $x2 = 1000 - int(rand(2000));
       my $y2 = 1000 - int(rand(2000));
       my $d2 = int(rand(500));
       my $z2 = sin($x2) * cos($y2) - $d2;

       $multilayer->input([ $x2 , $y2 , $z2 ]);
       $ret = $multilayer->calc_multi();
       say "!!!! CHECK: @{$ret->[-1]} hope: 01 -----"; 

       $retstring = join ("" , @{$ret->[-1]});
       push(@{$ver_chk} , 'ok' ) if $retstring eq '01';

       Logging("DEBUG: verification: $verification_count ");

       # classラベル毎にチェックして"okokok"であれば完了
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

 } # while verification

       $multilayer->dump_structure();


