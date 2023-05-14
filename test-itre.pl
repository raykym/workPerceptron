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


my $x = sequence(2,3,4,5,6,7,8,9);

my $pdlitre = Pdlitre_idx->new($x);

while ($pdlitre->finished) {

    say $pdlitre->multi_index();

    $pdlitre->itrenext();

}






package Pdlitre_idx;

# PDLを受けて、全体のインデックスをタプルで返す
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
       #  $self->{idx_list} = []; # perl配列は作成しない
       #$self->{idx_out} = []; # 出力用リファレンス
       $self->{list_cnt} = undef; # 出力インデックスの添字最大値
       $self->{idx_pdl} = undef; #作成したインデックスPDL
       $self->{idx_point} = 0; # インデックスの現在のポイント

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
    my $IDX_PDL = cat @axis;  #インデックスPDLを1次元で結ぶ
       $self->{list_cnt} = $PDL->nelem;
       $self->{list_cnt}--; # 添字なので-1
       $self->{idx_pdl} = $IDX_PDL;
       $self->{idx_point} = 0;

=pod    # perl配列に置き直すとメモリを食いすぎるので、再検討
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
=cut

    undef $IDX_PDL;
    undef @axis;
    undef @dims;
    undef $loop;
}

sub finished {
    my $self = shift;

    if ($self->{list_cnt} > $self->{idx_point} ) {
        return 1; # 偽 未了
    } elsif ($self->{list_cnt} <= $self->{idx_point}) {
        return 0; # 真 完了
    }
}

sub multi_index {
    my $self = shift;
    # itrenextを実行しないと更新されない

    return list($self->{idx_pdl}->range($self->{idx_point}));

}

sub itrenext {
    my $self = shift;
    # idx_pointを一つ増やす

    # $self->{idx_out} = shift(@{$self->{idx_list}});
    $self->{idx_point}++;
}

