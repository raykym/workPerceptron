#!/usr/bin/env perl
#
#
# 初期設定しているとdefinedでは判定出来ない
#
use strict;
use warnings;
use utf8;
use feature ':5.13';

binmode 'STDOUT' , ':utf8';

# 判定処理のおさらい

my $tmp = [];

my $test = "";

if (defined $tmp) {
   say "defined tmp";
} else {
   say "not defined";
}

if (@{$tmp}) {
   say "tmp in value";
} else {
   say "tmp is null";
}

if (defined $test) {
	say "test defined";
} else {
	say "test not defined";
}


use Data::Dumper;

my $out = [];
for (my $i=0; $i<=3; $i++) {

    push(@{$out->[$i]} , $i );
    push(@{$out->[$i]} , $i+1 );

}

print Dumper $out;


my $aaa = [];
my $bbb = [];

push(@{$aaa} , 'a');
push(@{$aaa} , 'b');
push(@{$aaa} , 4);
push(@{$aaa} , 8);

say @{$aaa};

push(@{$bbb} , $aaa);

say @{$bbb};



