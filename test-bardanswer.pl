#!/usr/bin/env perl

# bardに問いかけて、分岐処理を使わずにサブルーチンを切り替える方法

#use v5.32;
use utf8;
binmode 'STDOUT' , ':utf8';

$|=1;



sub foo {
  print "foo\n";
}

sub bar {
  print "bar\n";
}

my @subroutines = qw(foo bar);

my $index = 0;

while ($index < @subroutines) {
  &{$subroutines[$index]};
  $index++;
}
