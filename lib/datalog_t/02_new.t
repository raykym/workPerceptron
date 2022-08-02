#t/02_new.t
  
use strict;
use warnings;
use Test::More;
use Test::Exception;
use FindBin;
use lib "$FindBin::Bin/..";
use Datalog;

subtest 'no_args' => sub {
   ok( my $obj = Datalog->new , 'no agrgs');
};

subtest '$str is null' => sub {
    my $str = '';
    dies_ok( sub { my $obj= Datalog->new($str) } , 'null input');
};

subtest '$str' => sub {
    my $str = 'hogehoge';
    ok ( my $obj = Datalog->new($str) , 'file name is hogehoge');
};

subtest 'sqlite3　file' => sub {

    ok ( -f './multilayer.sqlite3' , 'file exist' );
    ok ( -f './hogehoge.sqlite3' , 'hogehoge file exist' );
};

unlink './multilayer.sqlite3';
unlink './hogehoge.sqlite3';

done_testing;
