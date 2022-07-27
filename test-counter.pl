#!/usr/bin/env perl
#
use strict;
use warnings;
use utf8;
use feature 'say';

binmode 'STDOUT' , ':utf8';

# while loop でカウンターを入れただけでメモリーをどの程度使うのか確認する。
# topコマンドで
#

$|=1;

srand();

sub Logging {
        my $logline = shift;
        my $dt = time();
        say "$dt | $logline";

        undef $dt;
        undef $logline;

        return;
}



my $count_flg = 1;
my $count = 0;

while ( $count_flg) {
    $count++;
    if ( $count > 20000 ) {
        Logging("count over 20000");
        $count_flg = 0;
    }

    my $learn_flg = 1;
    my $learn_cnt = 0;
    while ( $learn_flg ) {
        $learn_cnt++;
        if ( $learn_cnt > 20000 ) {
            Logging("learn_cnt over 20000");
            $learn_flg = 0;
        }

        for my $cnt ( 1 .. 100 ) {

            # ここはデータ生成のフィールド


        }


        # learning フィールド


    } # while learn_flg

    # checkフィールド

} # while $count_flg



