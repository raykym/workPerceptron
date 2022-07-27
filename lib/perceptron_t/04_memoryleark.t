use strict;
use warnings;
use Test::More;
use Test::Exception;
use FindBin;
use lib "$FindBin::Bin/..";
use Perceptron;

use Devel::Size;


subtest 'memory leark test' => sub {
    # 現実的にはsizeに変化があると表示されるので、履歴を見て判断するもの
    # proveではなく、perl 04_memoryleark.t で実行するもの

    my $learndata_ANDgate = [
                      {
                        class => 1 ,
                        input => [ 1 , 1 ]
                      },
                      {
                        class => -1 ,
                        input => [ -1 , 1 ]
                      },
                      {
                        class => -1 ,
                        input => [ 1 , -1 ]
                      },
                      {
                        class => -1 ,
                        input => [ -1 , -1 ]
                      },
                    ];

    my $learndata_sample = [];
    for my $count ( 1 .. 100 ) {
        my $x = int(rand(100));
        my $y = int(rand(100));
	my $sample = {};
	$sample->{class} = 1;
	$sample->{input} = [ $x , $y ];

	push(@{$learndata_sample} , $sample);

	my $z = - int(rand(100));
	my $o = - int(rand(100));
	my $sample2 = {};
	$sample2->{class} = -1;
	$sample2->{input} = [ $z , $o ];

        push(@{$learndata_sample} , $sample2);
    }

    my $unit = Perceptron->new();

    my $total_size_init = Devel::Size::total_size($unit);
    print "init size: $total_size_init\n";

    my $loop = 1;
    my $loop_cnt = 0;
    my $total_size_point = 0;
    my $total_size_point_cp = 0;
    while ($loop) {
         eval { $unit->learn_simple($learndata_sample) }; # 基本的に一度で成功しないので、エラーをキャッチしてループさせている
	 # SPVMが原因ではなく、forループに参照渡しを読んで、そのままになるとリークが起きる。
	 #eval { $unit->dummy_method($learndata_sample) }; # 常にエラーになる ->リークなし
	 #eval { $unit->waitsinit() };  # 常に成功するのでloopでカウントされる　->リークなし
	 #eval {  $unit->input($learndata_sample->{input}); $unit->waitsinit(); $unit->calcReLU(); }; # ->リークなし
	 #eval {  $unit->input($learndata_sample->{input}); $unit->waitsinit(); $unit->calcStep(); }; # ->リークなし
        if ($@ ) {
             # on errror 
	     #  print "on error!\n";
            $total_size_point = Devel::Size::total_size($unit);
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

    my $total_size_end = Devel::Size::total_size($unit);
    print "end size: $total_size_end\n";

    is ( $loop , 0 , 'loop end' );

};

done_testing;
