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

# 引数にhparamsのファイル名を読み込む

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

for my $idx ( 0 .. 10 ) {

    my $pice_PDL = $train_x->range($idx);
    my $pice_t = $train_t->range($idx);

    my $res = $hparams->{network}->predict($pice_PDL);

    say "$pice_t : $res";

}



