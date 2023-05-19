#!/usr/bin/env perl
#
# perl AI深層学習入門サイトからコピー
# MNISTのダウンロードバイナリーをPDLに置き換える

use strict;
use warnings;
use FindBin;
use feature 'say';

use Data::Dumper;
use PDL;
#use PDL::IO::FastRaw;
use PDL::IO::FlexRaw;
use PDL::NiceSlice;

# MNIST画像情報を読み込む
my $mnist_image_file = "$FindBin::Bin/MNIST/train-images-idx3-ubyte";
#my $mnist_image_file = "$FindBin::Bin/MNIST/t10k-images-idx3-ubyte";

open my $mnist_image_fh, '<', $mnist_image_file
  or die "Can't open file $mnist_image_file: $!";

# マジックナンバー
my $image_buffer;
read($mnist_image_fh, $image_buffer, 4);
my $magic_number = unpack('N1', $image_buffer);
if ($magic_number != 0x00000803) {
  die "Invalid magic number expected " . 0x00000803 . "actual $magic_number";
}

# 画像数
read($mnist_image_fh, $image_buffer, 4);
my $items_count = unpack('N1', $image_buffer);

# 画像の行ピクセル数
read($mnist_image_fh, $image_buffer, 4);
my $rows_count = unpack('N1', $image_buffer);

# 画像の列ピクセル数
read($mnist_image_fh, $image_buffer, 4);
my $columns_count = unpack('N1', $image_buffer);

=pod
# 画像の読み込み
my $image_data;
my $all_images_length = $items_count * $rows_count * $columns_count;
my $read_length = read $mnist_image_fh, $image_data, $all_images_length;
unless ($read_length == $all_images_length) {
  die "Can't read all images";
}
=cut

my @train_x;  # トレーニングデータ　PDLのperl配列

my $offset = 16;
for my $cnt ( 1 .. $items_count ) {
    seek($mnist_image_fh , $offset , 0 ); 
    push (@train_x , readflex $mnist_image_fh , [{NDims => 2 , Dims => [ $rows_count , $columns_count ] , Type => 'byte'}]) ;
    $offset += ($rows_count * $columns_count);

}


# 画像情報
my $image_info = {};
$image_info->{items_count} = $items_count;
$image_info->{rows_count} = $rows_count;
$image_info->{columns_count} = $columns_count;
$image_info->{data} = \@train_x;

#say Dumper $image_info;

say $image_info->{items_count};
say $image_info->{rows_count};
say $image_info->{columns_count};

#say $train_x[2]->shape;
#for my $i ( 0 .. 5 ) {
#    say $train_x[$i];
#}

# 1次元に直す
for my $img (@train_x) {
    $img = $img->flat;
}

my $train_x_2D = cat @train_x ;  # ( 784 , 60000)
say $train_x_2D->shape;
    $train_x_2D = $train_x_2D->transpose; #次元入れ替え
say $train_x_2D->shape;
    $train_x_2D = $train_x_2D->reshape(60000,28,28);
say $train_x_2D->shape;
say $train_x_2D->range(0);

exit;

say "";


# Label data

my $mnist_labels_file = "$FindBin::Bin/MNIST/train-labels-idx1-ubyte";
# $mnist_image_file = "$FindBin::Bin/MNIST/t10k-labels-idx1-ubyte";

open  my $mnist_labels_fh, '<', $mnist_labels_file
  or die "Can't open file $mnist_labels_file: $!";

# マジックナンバー
   my $labels_buffer;
read($mnist_labels_fh, $labels_buffer, 4);
   $magic_number = unpack('N1', $labels_buffer);
if ($magic_number != 0x00000801) {
  die "Invalid magic number expected " . 0x00000801 . "actual $magic_number";
}

# ラベル数
read($mnist_labels_fh, $labels_buffer, 4);
my $labels_count = unpack('N1', $labels_buffer);

=pod
# ラベルの読み込み
  #$image_data;
   $all_images_length = $labels_count;
   $read_length = read $mnist_image_fh, $image_data, $all_images_length;
unless ($read_length == $all_images_length) {
  die "Can't read all images";
}
=cut;

my @train_label; #perl配列にPDLが入る
 $offset = 8;
for my $cnt ( 1 .. $labels_count ) {
    seek($mnist_labels_fh , $offset , 0 ); 
    push ( @train_label , readflex $mnist_labels_fh , [{Dims => [ 1 ] , Type => 'byte'}] ); 
    $offset++;
}

say $labels_count;

say $train_label[2]->shape;
#for my $i ( 0 .. 5 ) {
#    say $train_label[$i];
#}

my $train_label_2D = cat @train_label;

say $train_label_2D->shape;
$train_label_2D = $train_label_2D->transpose;
say $train_label_2D(0);

