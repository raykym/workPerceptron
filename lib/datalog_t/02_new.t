#t/02_new.t
  
use strict;
use warnings;
use Test::More;
use FindBin;
use lib "$FindBin::Bin/..";
use Datalog;

subtest 'no_args' => sub {
   my $obj = Datalog->new;
   isa_ok $obj, 'Datalog';
};

subtest '$str is null' => sub {
    my $str = '';
    my $obj = Datalog->new($str);
    isa_ok $obj, 'Datalog';
};

subtest '$str' => sub {
    my $str = 'hogehoge';
    my $obj = Datalog->new($str);
    isa_ok $obj, 'Datalog';
};

subtest 'sqlite3　file' => sub {

    ok ( -f './multilayer.sqlite3' , 'file exist' );
};

unlink './multilayer.sqlite3';

done_testing;
