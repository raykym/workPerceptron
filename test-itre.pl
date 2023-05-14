#!/usr/bin/env perl
#
# PDLを受けて、インデックスの配列を返す関数
#

use strict;
use warnings;
use utf8;
use feature 'say';

binmode 'STDOUT' , ':utf8';

use PDL;
use PDL::NiceSlice;
use PDL::Core ':Internal';


my $x = sequence(2,3,4,5,8);

my $pdlitre = Pdlitre_idx->new($x);

while ($pdlitre->finished) {

    say $pdlitre->multi_index();

    $pdlitre->itrenext();

}






package Pdlitre_idx;

# PDLを受けて、全体のインデックスをタプルで返す
# 100万要素位になると8GBの4%程度使ってしまう
# 
use PDL;
use PDL::NiceSlice;
use PDL::Core ':Internal';

sub new {
    my $proto = shift;
    my $class = ref $proto || $proto;

    my $self = {};
       $self->{pdl} = shift;  # pdlを想定
       $self->{pdl} = topdl($self->{pdl});
       $self->{idx_list} = [];
       $self->{idx_out} = undef;

    bless $self , $class;

    $self->_init();

    return $self;
}

sub _init {
    my $self = shift;
    # PDLのインデックス perl配列を作成する
    # 大文字変数はPDL 小文字はperl変数でリファレンスとしてPDLが入っているケースが在る


    my $PDL = $self->{pdl};
    my @dims = $PDL->dims; # shape

    my $loop = 0;
    my @axis = (); # perl array in PDL
    while ( $loop <= $#dims ) {   # 次元の数だけループする
       # loopはaxisに対応する
       $axis[$loop] = axisvals($PDL, $loop); # index PDLを次元毎に作成
       $axis[$loop]->reshape($axis[$loop]->nelem); # 1次元にreshape
       $loop++;
    }
    my $IDX_PDL = cat @axis;  #インデックスPDLを次元で結ぶ

    # perl配列に置き直すとメモリを食いすぎるので、再検討
    my $list_cnt = $PDL->nelem; # 入力PDLの要素数
       $list_cnt--; # 添字なので-1
    my $cnt = 0;
    while ( $cnt <= $list_cnt ) {
        my @idx = list($IDX_PDL->range($cnt)); # ( 2,3,0,・・・) の形式
        push(@{$self->{idx_list}} , \@idx );
        $cnt++;
    }
    # {idx_list}にインデックスがリストされる
    #
    # 最初の1個だけはセットしておく
    $self->{idx_out} = shift(@{$self->{idx_list}});


    undef $IDX_PDL;
    undef @axis;
    undef @dims;
    undef $list_cnt;
    undef $loop;
}

sub finished {
    my $self = shift;

    if (@{$self->{idx_list}}) {
        return 1; # 偽 未了
    } else {
        return 0; # 真 完了
    }
}

sub multi_index {
    my $self = shift;
    # itrenextを実行しないと更新されない

    return @{$self->{idx_out}}; # 配列をそのまま返す
}

sub itrenext {
    my $self = shift;

    $self->{idx_out} = shift(@{$self->{idx_list}});
}

