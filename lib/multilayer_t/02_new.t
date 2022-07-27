#t/02_new.t
  
use strict;
use warnings;
use Test::More;
use FindBin;
use lib "$FindBin::Bin/..";
use Multilayer;

subtest 'no_args' => sub {
   my $obj = Multilayer->new;
   isa_ok $obj, 'Multilayer';
};

subtest '$str is null' => sub {
    my $str = '';
    my $obj = Multilayer->new($str);
    isa_ok $obj, 'Multilayer';
};

subtest '$str' => sub {
    my $str = 'hogehoge';
    my $obj = Multilayer->new($str);
    isa_ok $obj, 'Multilayer';
};

done_testing;
