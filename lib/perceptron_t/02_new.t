#t/02_new.t
  
use strict;
use warnings;
use Test::More;
use FindBin;
use lib "$FindBin::Bin/..";
use Perceptron;

subtest 'no_args' => sub {
   my $obj = Perceptron->new;
   isa_ok $obj, 'Perceptron';
};

subtest '$str is null' => sub {
    my $str = '';
    my $obj = Perceptron->new($str);
    isa_ok $obj, 'Perceptron';
};

subtest '$str' => sub {
    my $str = 'hogehoge';
    my $obj = Perceptron->new($str);
    isa_ok $obj, 'Perceptron';
};

done_testing;
