#!/usr/bin/env perl
#
# MnistLoadモジュールを試す
# PDLをパッケージに重複入力した場合、PDLが維持できないケースを検証する
# 関数の引数でPDLを渡すとメソッドが使えないものが在る。

use strict;
use warnings;
use utf8;
use feature 'say';
use v5.32;

binmode 'STDOUT' , ':utf8';

$|=1;

use PDL;
use PDL::Core ':Internal';
use PDL::NiceSlice;

use FindBin;
use lib "$FindBin::Bin/../lib";
use MnistLoad;

my ( $train_x , $train_l , $test_x , $test_l ) = MnistLoad::mnistload();

#$train_x = topdl($train_x);
#$train_l = topdl($train_l);
#$test_x = topdl($test_x);
#$test_l = topdl($test_l);

say $train_x->shape;
say $train_l->shape;
say $test_x->shape;
say $test_l->shape;

    $train_x = $train_x->reshape(60000,28,28);
say $train_x->range(0);
say $train_l(0);

    $test_x = $test_x->reshape(10000,28,28);
say $test_x->range(0);
say $test_l(0);

    $train_x = $train_x->reshape(60000,784);
    $test_x = $test_x->reshape(60000,1);

    say $train_x->shape;
    say $train_x->range(0);

    say $test_x->shape;
    say $test_x->range(0);

    say "ref train_x";
    say ref $train_x;
    say \$train_x;

    my $pkg1 = Pkg1->new($train_x , $train_l);

    $pkg1->doSomething;


package Pkg1;

use PDL;
use PDL::Core ':Internal';
use PDL::NiceSlice;
use PDL::Slices;   # doSomethingでindex1dを使う場合に必要になる

sub new {
    my $proto = shift;
    my $class = ref $proto || $proto;
    
    my $self = {};
    bless $self , $class;

    my ($train_x , $train_l ) = @_;

    $self->{train_x} = $train_x;
    $self->{train_l} = $train_l;

    #say "pkg1 new";
    #say $self->{train_x}->info;

    return $self;
}

sub doSomething {
    my $self = shift;
    
    say "method doSomething";
    say $self->{train_x}->shape;
    say $self->{train_l}->dims;
    say \$self->{train_x};

    my @tmp = ( 1 );
    my $pdl = pdl(@tmp);
    say $pdl;
    # ->index1dはPDL::Slicesに含まれるが、関数引き渡しではPDL::Core以外は引き継げないらしい。
    # ここのパッケージ内で初期宣言されるとPDL::Slicesもまとめられるが、引き継ぎでは機能を持ち合わせていない。。。。。
    # mainスコープでは宣言は省略されていたが、関数に引き継いだ場合は個別にuse宣言しないと使えない。。。。
    #say $self->{train_x}->index1d($pdl);

    # $tain_x->index1d()ならパッケージ内にuse宣言が無くても利用が可能。
    # PDLはmain関数でグローバル変数として宣言したほうがメモリー効率は良いのだろうけど、
    # 処理毎に分離して考えるなら、core以外は都度useするしか無いか。。。

}
