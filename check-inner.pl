#!/usr/bin/env perl
#
# SPVMのUtil.spvm onedlinersum の動作がおかしいみたいなのでチェックする。
#

use strict;
use warnings;
use feature 'say';

use utf8;

binmode 'STDOUT' , ':utf8';

$|=1;

use FindBin;
use lib "$FindBin::Bin/lib";
use SPVM 'Util';

use PDL;

# テストデータ作成

my $datalist = [];  # 2次元
for my $i ( 1 .. 10 ) {
    my $waits = [];
    my $inputs = [];

    for my $j ( 1 .. 10 ) {
        my $wait = rand( 1 / 10 ); # node 10個を想定
        my $input = rand(2);

	push( @{$waits} , $wait );
	push( @{$inputs} , $input );
    }

    push(@{$datalist} , { waits => $waits , inputs => $inputs } );
}

my $perl_calc = []; # 1次元
my $spvm_calc = [];
my $pdl_calc = [];

# データループ　10x10
for my $sample ( @{$datalist} ) {

    # perl calc
    my @input = @{$sample->{inputs}};
    my @waits = @{$sample->{waits}};

    my $p_sum = undef;
    for (my $i=0; $i<= $#input ; $i++ ) {
        $p_sum += $input[$i] * $waits[$i] ;
    } 
    push (@{$perl_calc} , $p_sum );

    # SPVM calc

    my $s_sum = undef;

    $s_sum = SPVM::Util->onedinnersum( $sample->{inputs} , $sample->{waits} );

    push(@{$spvm_calc} , $s_sum);


    # PDL calc
    my $i = pdl $sample->{inputs};
    my $w = pdl $sample->{waits};

    my $pdl_sum = sum( $i * $w ); 
    push(@{$pdl_calc} , $pdl_sum);

} # for sample


# 確認
#
say "Data list";
for my $sample (@{$datalist}) {
    say @{$sample->{inputs}};
    say "";
    say @{$sample->{waits}};
    say "-----";
}

say "";
say "perl | SPVM | PDL";
for ( my $i = 0 ; $i<= 9 ; $i++ ) {

    say "$perl_calc->[$i] | $spvm_calc->[$i] | $pdl_calc->[$i]";

}

