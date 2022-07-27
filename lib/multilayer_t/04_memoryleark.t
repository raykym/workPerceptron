use strict;
use warnings;
use Test::More;
use Test::Exception;
use FindBin;
use lib "$FindBin::Bin/..";
use Perceptron;
use Multilayer;

use Devel::Size;



subtest 'memory leark test' => sub {
    # 現実的にはsizeに変化があると表示されるので、履歴を見て判断するもの
    # proveではなく、perl 04_memoryleark.t で実行するもの
    # topコマンドで大雑把に状況を見る

    my $dumpData = require '../../dump_structure.txt';

    my $multilayer = Multilayer->new();

    $multilayer->layer_init($dumpData->{layer_init}); 
    $multilayer->takeover($dumpData); 

    my $total_size_init = Devel::Size::total_size($multilayer);
    print "init size: $total_size_init\n";

    my $loop = 1;
    my $loop_cnt = 0;
    my $total_size_point = 0;
    my $total_size_point_cp = 0;
    while ($loop) {

    ##########
    my $learndata_sample = [];
    for my $count ( 1 .. 100 ) {
        my $x = int(rand(100));
        my $y = int(rand(100));
        my $x1 = int(rand(100));
        my $y1 = int(rand(100));
	my $sample = {};
	$sample->{class} = [ 1 , 0 ];
	$sample->{input} = [ $x , $y , $x1 , $y1 ];

	push(@{$learndata_sample} , $sample);

	my $z = - int(rand(100));
	my $o = - int(rand(100));
	my $z1 = - int(rand(100));
	my $o1 = - int(rand(100));
	my $sample2 = {};
	$sample2->{class} = [ 0 , 1 ];
	$sample2->{input} = [ $z , $o , $z1 , $o1 ];

        push(@{$learndata_sample} , $sample2);
    }
    ##########


         eval { $multilayer->learne($learndata_sample) }; # リークあり?
	 #eval { $multilayer->layer_init($learndata_sample->{layer_init}) }; # リークなし
	 #eval { $multilayer->disp_waits() }; # リークなし
	 #eval { $multilayer->stat('learned'); $multilayer->input($learndata_sample->{input}); $multilayer->calc_multi() }; # リークなし
        if ($@ ) {
             # on errror 
	     #  print "on error!\n";
            $total_size_point = Devel::Size::total_size($multilayer);
	    if (( $total_size_point > $total_size_point_cp ) || ( $total_size_point < $total_size_point_cp ) ) {
                print "point size: $total_size_point\n";
            } 
        } else {
	    $loop_cnt++;
	    if ($loop_cnt > 100 ) {
	        # loop end
                $loop = 0;
            }
        }
	$total_size_point_cp = $total_size_point;
    } # while

    my $total_size_end = Devel::Size::total_size($multilayer);
    print "end size: $total_size_end\n";

    is ( $loop , 0 , 'loop end' );

};

done_testing;
