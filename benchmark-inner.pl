#!/usr/bin/env perl

# 一次元配列の内積を計算する処理を比較する。
# 1.perl code
# 2.PDL
# after 
# 3.SPVM

use strict;
use warnings;
use utf8;

binmode 'STDOUT' , ':utf8';

$|=1;

use Benchmark;
use PDL;
use Clone;

use FindBin;
use lib "$FindBin::Bin/lib";
use SPVM 'Util';

=pod
 perl for構文とPDLを比較した

 100程度の入力ではperlの方が早い
 10000の入力ではPDLが早い
 320まではperlが早い

 入力が320項目以下の場合はfor構文で処理しても良い
 それ以上の項目になる場合はPDLか、SPVMを検討する
 ----
 perlの処理を簡単に書き直したところ、50000000以上でPDLがようやく上回るようになった
 500万項目のインプットって、perlでも十分なのでは。。。
 入力が整数の場合はこのとおり

 ---
 入力を整数から少数に変更して試したところ
     [ 1 .. 5000000] から rand(1)で処理した
 280以上でPDLが早い

---
 SPVMを加えると、
 SPVMが圧倒的

=cut

my $count = 270;

my $input = [];
my $waits = [];
my @input = @{$input};
my @waits = @{$waits};
my $bias = 555;

srand(12345);

my $tic = 0;

while ( $tic <= $count ) {

    my $rand = rand(5000000);
    my $rand2 = rand(1);
    push(@input , $rand);
    push(@waits , $rand2);

    $tic++;
}


Benchmark::cmpthese(-1, {
    'perl for' => sub {
=pod
        my @mul = ();
        for (my $i=0; $i<=$#input ; $i++) {
            push(@mul , ($input[$i] * $waits[$i]) );
        }
        my $sum = 0;
        for (my $i=0; $i<=$#mul; $i++) {
            $sum = $sum + $mul[$i];
        }
=cut
	my $sum = 0;
	for (my $i=0; $i<=$#input; $i++) {
            $sum += ( $input[$i] * $waits[$i] );
	}

        if ( $sum > $bias ){
            return 1;
        } elsif ( $sum < $bias ) {
            return 0;
        }

    },

    'PDL' => sub {
        my $i = pdl $input;
        my $w = pdl $waits;
        my $sum = sum($i * $w);

        if ($bias > $sum) {
            return 1;
        } elsif ($sum < $bias) {
            return 0;
        }

    },

    'SPVM' => sub {

        my $sum = SPVM::Util->onedinnersum( $input , $waits );

        if ( $sum > $bias ){
            return 1;
        } elsif ( $sum < $bias ) {
            return 0;
        }
    },

	});
