#!/usr/bin/env perl
#
# 非復元抽出のためにインデックスを作成する手法
#

use v5.32;
use utf8;
use List::Util qw/uniq/;

binmode 'STDOUT' , ':utf8';

my @array = ();

for my $i ( 0 .. 40401) {
    push(@array , [ $i , rand(100000) ] );
}

my @array_sort = sort { $a->[1] <=> $b->[1] } @array;

my @index = ();
for my $idx ( 0 .. 9999 ) {
    push(@index , $array_sort[$idx]->[0]);
}

my $count = uniq @index;

say $count;
