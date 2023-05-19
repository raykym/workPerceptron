#!/usr/bin/env perl
#

use strict;
use warnings;
use utf8;
use v5.32;

binmode 'STDOUT' , ':utf8';

$|=1;

use Data::Dumper;
use PDL;
use PDL::Core ':Internal';
use PDL::IO::GD;

# MNISTディレクト下のデータにアクセスして、読み込み可能な状態にする

chdir("./MNIST/train/");
my @files = glob "*.png";

my @train_list; # { filename => XXXX , number => X } の形式で入る
for my $file ( @files) {
    my $num = substr($file , 10 , 1); # ファイル名の末尾を取得
    my $filename = "./MNIST/train/";
       $filename .= $file;
    push (@train_list, { filename => $filename , number =>  $num }) ;
}

#for my $list (@train_list){
#    say Dumper $list;
#}
say "$train_list[0]->{filename}";

my $gd = PDL::IO::GD->new("$train_list[0]->{filename}");
my $im_pdl = $gd->to_pdl;

#my $im = read_png("$train_list[0]->{filename}");
#my $im_pdl = $im->to_pdl;
#print $im_pdl;
