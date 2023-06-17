#!/usr/bin/env perl
#
# *.hparamsを読み込んで使う
#

use v5.32;
use utf8;

binmode 'STDOUT' , ':utf8';
binmode 'STDIN' , ':utf8';

$|=1;

use PDL;
use PDL::Core ':Internal';
use PDL::NiceSlice;

use Storable qw / store retrieve /;
use PDL::IO::Storable;

use FindBin;
use lib "$FindBin::Bin::../lib";
use MultiLayerNet;
use Adam_optimizer;
use MnistLoad;
use Ml_functions;

# 引数にhparamsのファイル名を読み込む

if (! defined $ARGV[0] ) {
    say "Usage: mnist-load.pl 'finename'";
    exit;
}


# MNISTファイルのロード
my ($train_x , $train_t , $test_x , $test_t ) = MnistLoad::mnistload();


my $hparams = {};
   $hparams = retrieve("$ARGV[0]");

   for my $key (keys %{$hparams} ) {
       if ( ! defined $hparams->{$key} ) {
           die "Data error!!!";
       }
       if ( $hparams->{$key} =~ /^[0-9]+$|^[0-9]+\.[0-9]+$/ ) {
           say "$key : $hparams->{$key}"; 
       } else {
           # 数値で無ければスルー
           next;
       }
   }

my $cnt = 0;
#my $offset = int(rand(9990));

#for my $idx ( $offset .. 10 + $offset ) {
for my $idx ( 0 .. 9999 ) {

    #my $pice_PDL = $test_x->range($idx);
    #my $pice_t = $test_t->range($idx);
    my $pice_PDL = $test_x->index($idx);
    my $pice_t = $test_t->index($idx);
    my @tmp2 = list($pice_t);

    my $res = $hparams->{network}->predict($pice_PDL);

    my $hotone = Ml_functions::argmax($res);

    my $one_idx = vsearch( 1 , $hotone , {mode => 'match'});
    my @tmp = list($one_idx); # 一致する値がないとここでは-11が返る???

    $cnt++ if $tmp[0] == $tmp2[0];

        # ラベル : 計算結果 : hotone表示
    say "$pice_t : $res : $hotone : @tmp"  if $tmp[0] == $tmp2[0];

}

    say "$cnt / 10000";

