use strict;
use warnings;
use Test::More;
use Test::Exception;
use FindBin;
use lib "$FindBin::Bin/..";
use Datalog;

use Data::Dumper;


subtest 'methods check' => sub {
    my $obj = Datalog->new;

    can_ok($obj , qw/ new _init addlog /);

};

subtest 'addlog method input check' => sub {
    my $obj = Datalog->new;
    my $array = [
                    {
                        bias => [] ,
                        waits => [ 1 , 2, 3 , 4 , 5 ],
		    },
                    {
                        waits => [ 0.1 , 0.23 , 0.93 ],
			bias => [],
		    }
               ];
    my $array_strings = Dumper $array;

    ok ( $obj->addlog($array_strings) , 'input array' );

    my $hash = {
	         waits => 'array data object',
		 bias => 'array data object',
	       };
    my $hash_strings = Dumper $hash;

    ok ( $obj->addlog($hash) , 'input hash' );

    my $scalar = 20;

    ok ( $obj->addlog($scalar) , 'input scalar' );

};

unlink './multilayer.sqlite3';

done_testing;






=pod
subtest 'waits method input check' => sub {
    my $obj = Perceptron->new;

    my $array = [ 1 , 2 , 3 ];
    my $hash = { 1 => 1 , 2 => 2 , 3 => 3 };
    my $scalar = 0.23;

    ok ($obj->waits($array) =~ /ARRAY/ , 'input refARRAY');
    dies_ok( sub { $obj->waits($hash) } , 'input refHASH die');
    isnt( $obj->waits(%{$hash}) , ref , 'input HASH maybe OUT but through');
    ok ( $obj->waits($scalar) =~ /ARRAY/ , 'input scaler is accept but ...');

};

subtest 'input method check' => sub {
    my $obj = Perceptron->new;

    my $array = [ 1 , 2 , 3 , 4 ];
    my $hash = { 1 => 1 , 2 => 2 , 3 => 3 };

    ok ( $obj->input($array) =~ /ARRAY/ , 'input refARRAY');
    dies_ok ( sub { $obj->input($hash) } , 'input refHASH die');
    ok ( $obj->input(@{$array}) =~ /ARRAY/ , 'input ARRAY' );
    ok ( $obj->input(%{$hash}) =~ /ARRAY/ , 'input HASH maybe OUT but through...');

};

subtest 'calc method & clacReLU calcStep check' => sub {
    my $obj = Perceptron->new;

    #長さの違う入力を弾く

    my $waits = [ 1 , 2 , 3 , 4 , 5 ];
    my $input = [ 1 , 2 , 3 , 4 ];

    $obj->waits($waits);
    $obj->bias(2);
    $obj->input($input);

    dies_ok ( sub { $obj->calc() } , 'calc method input and waits input miss match!');
    dies_ok ( sub { $obj->calcReLU() } , 'calcReLU method input and waits input miss match!');
    dies_ok ( sub { $obj->calcStep() } , 'calcStep method input and waits input miss match!');

};

subtest 'waitsinit methos check' => sub {
    my $obj = Perceptron->new;

    dies_ok( sub { $obj->waitsinit() } , ' no input , no params , no learn_input' );

    my $array = [ 1 , 2 , 3 , 4 , 5 ];
    $obj->input($array);
    $obj->waitsinit();
    ok( $obj->waits() , 'first input set ' );

    #leaarn_inputはlearn_simpleと連結の為、パス
    #

    # param set
    $obj = Perceptron->new;
    $obj->waitsinit(4);
    my $res = $obj->waits();
    my @res = @{$res};
    my $bias = $obj->bias();
    ok( $#res == 4 , 'one param setup' );

    $obj = Perceptron->new;
    dies_ok( sub { $obj->waitsinit($array) } , 'not number' );

    $obj = Perceptron->new;
    $obj->waitsinit(4 , 3 );
    $res = $obj->waits();
    @res = @{$res};
    $bias = $obj->bias();
    ok ( $#res == 4 , 'two param setup' );

};

subtest 'learn_rate method check' => sub {
    my $obj = Perceptron->new;

    ok( $obj->learn_rate(0.23) , 'input right number' );

    dies_ok ( sub { $obj->learn_rate(1.8) } , 'input rate 0.0~1.0' );

};

subtest 'learn_simple method ' => sub {
	# 確率で失敗する！！！！！！！！！

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

    my $unit = Perceptron->new();

    ok ( $unit->learn_simple($learndata_ANDgate) , 'learn_simple method do' );

};
=cut

