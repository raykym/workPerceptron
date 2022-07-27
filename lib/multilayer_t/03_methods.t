use strict;
use warnings;
use Test::More;
use Test::Exception;
use FindBin;
use lib "$FindBin::Bin/..";
use Perceptron; 
use Multilayer;


subtest 'methods check' => sub {
    my $obj = Multilayer->new;

    can_ok($obj , qw/ new layer_init takeover disp_waits dump_structure learn stat input calc_multi /);

};

subtest 'layer_init method check' => sub {
    my $obj = Multilayer->new;

    my $structure = {
                      layer_member  => [ 1 , 1 ],
                      input_count => 3 ,
                      learn_rate => 0.34
                    };

    dies_ok( sub { $obj->layer_init() } , 'no param die' );

    my $array = [ 1 , 2 , 3 , 4 , 5 ];
    dies_ok( sub { $obj->layer_init($array) } , 'no HASH reference' );

    my $scalar = 1000;
    dies_ok( sub { $obj->layer_init($scalar) } , 'param is scalar' );

    my $stat = $obj->stat();
    if ($stat eq "" ) {
        print "stat is [ $stat ] ..\n";
    } 

    $obj->layer_init($structure);
    $stat = $obj->stat();
    is( $stat , 'layer_inited' , 'layer_init done' );

};

subtest 'disp_waits  method input check' => sub {
    my $obj = Multilayer->new;

    my $structure = {
                      layer_member  => [ 1 , 1 ],
                      input_count => 3 ,
                      learn_rate => 0.34
                    };
    $obj->layer_init($structure);

    # 標準出力を変数に取得して、チェックする
    my $stdout;
    open (my $tmpout , '>&' , STDOUT);
    close STDOUT;
    open STDOUT , '>' , \$stdout;
    $obj->disp_waits();

    close STDOUT;
    open STDOUT , '>&' , $tmpout;
    close $tmpout;

    print $stdout;
    print "\n";

    ok( $stdout , 'output waits numbers' );

};

subtest 'input method check' => sub {
    my $obj = Multilayer->new;

    my $array = [ 1 , 2 , 3 , 4 ];
    my $hash = { 1 => 1 , 2 => 2 , 3 => 3 };

    ok ( $obj->input($array) =~ /ARRAY/ , 'input refARRAY');
    dies_ok ( sub { $obj->input($hash) } , 'input refHASH die');
    dies_ok ( sub {$obj->input(@{$array}) } , 'input ARRAY die' );
    dies_ok ( sub { $obj->input(%{$hash}) } , 'input HASH die');

};

subtest 'stat method check' => sub {
    my $obj = Multilayer->new;

    # statは文字列を入れて状態を記録するだけなので、特に制限はない

    my $stat = $obj->stat();
    ok($stat eq "" , 'init stat is null');

    ok($obj->stat('learn') , 'input is word');
    
    $stat = $obj->stat();
    is( $stat , 'learn' , 'input word is return');

};

subtest 'dump_structure methos check' => sub {
    my $obj = Multilayer->new;

    my $structure = {
                      layer_member  => [ 1 , 1 ],
                      input_count => 3 ,
                      learn_rate => 0.34
                    };

    $obj->layer_init($structure);

    my $check = $obj->dump_structure('check');

    ok( $check =~ /HASH/ , 'dumpstructure check output');

    if ( -f './dump_structure.txt') {
        die "dump_structure.txt exist ...";
    } else {
        $obj->dump_structure();
    }    

    ok( -f './dump_structure.txt' , 'dump file makeing' );

};

subtest 'takeover method check' => sub {
    my $obj = Multilayer->new;

    my $dumpdata;
    if ( -f './dump_structure.txt' ) {
        $dumpdata = require './dump_structure.txt';
    } else {
        die "dump file not found!";
    }

    ok( $obj->layer_init($dumpdata->{layer_init}) , 'restructure to dumpdata');

    $obj->takeover($dumpdata);

    my $nowstructure = $obj->dump_structure('check');

    # 同じ位置のデータを比較する
    my $dumpstrings = join ("" , @{$dumpdata->{waitsdump}->{0}->{0}});
    my $nowstrings = join ("" , @{$nowstructure->{waitsdump}->{0}->{0}});

    is( $dumpstrings , $nowstrings  , 'load waits and bias data');

    unlink './dump_structure.txt';

};

subtest 'calc_multi method check ' => sub {
    my $obj = Multilayer->new;

    my $structure = {
                      layer_member  => [ 1 , 1 ],
                      input_count => 3 ,
                      learn_rate => 0.34
                    };

    $obj->layer_init($structure);

    dies_ok( sub { $obj->calc_multi('hogehoge') } , 'argument miss' );

    dies_ok( sub { $obj->calc_multi() } , 'stat not learned...');

};

subtest 'learn method ' => sub {

    my $obj = Multilayer->new;

    my $structure = {
                      layer_member  => [ 1 , 1 ],
                      input_count => 3 ,
                      learn_rate => 0.34
                    };

    dies_ok( sub { $obj->learn() } , 'stat is not layerinited' );

    $obj->layer_init($structure);

    dies_ok( sub { $obj->learn() } , 'learn data not set' );

    my $hash = { data => 'aaa' , node => 2 , stat => 'stand' };

    dies_ok( sub { $obj->learn($hash) } , 'learn data not refARRAY' );

};


done_testing;
