package MnistLoad;
#
# MNISTのダウンロードバイナリーをPDLに置き換える
# トレーニングデータ　PDL shape (60000,784)
# トレーニングラベル (60000,1)
# テストラベル (10000,784)
# テストラベル (10000,1)
#
# 標準化、ホットワンへの書き換えはここでは行わない
# フラットは行う
#
# ホットワンへの変換はchg_hotone関数
# 標準化 normalize関数

use feature 'say';

use Data::Dumper;
use PDL;
use PDL::Core ':Internal';
use PDL::IO::FlexRaw;
use PDL::NiceSlice;

sub mnistload {

    # ダウンロード済みのバイナリーファイルのパス
    my @mnist_imagefile = ( "/home/debian/perlwork/work/workPerceptron/MNIST/train-images-idx3-ubyte" , "/home/debian/perlwork/work/workPerceptron/MNIST/t10k-images-idx3-ubyte" ); 

    my @image_PDL = (); # ( trainのPDL , testのPDL)で格納される想定

    while ( my $filename = shift @mnist_imagefile ) {
        push (@image_PDL , &imageload($filename));
    }

    # イメージロードサブルーチン
    sub imageload {
        my $mnist_image_file = shift;

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

	undef $image_buffer;

        my @image_x; # PDLのperl配列

        my $offset = 16;
        for my $cnt ( 1 .. $items_count ) {
            seek($mnist_image_fh , $offset , 0 );
            push (@image_x , readflex $mnist_image_fh , [{NDims => 2 , Dims => [ $rows_count , $columns_count ] , Type => 'byte'}]);
            $offset += ($rows_count * $columns_count);
        }

	close $mnist_image_fh;

	# 1次元に変換 flat化
	for my $img (@image_x) {
            $img = $img->flat;
	}

        my $image_x_2D = cat @image_x; # 一つのPDLにまとめる (784 , 60000)
	   $image_x_2D = $image_x_2D->transpose; # (60000,784)

	undef @image_x;

=pod	# 以下のようにすると3次元でアクセスできる
	$image_x_2D = $image_x_2D->reshape(60000,28,28);
        say  $train_x_2D->range(0); # で2Dデータにアクセス可能
=cut

        $image_x_2D = convert($image_x_2D , double);

        return $image_x_2D;

    } # sub imageload


    # Label data
    # ダウンロード済みのバイナリーファイルのパス
    my @mnist_labelfile = ( "/home/debian/perlwork/work/workPerceptron/MNIST/train-labels-idx1-ubyte" , "/home/debian/perlwork/work/workPerceptron/MNIST/t10k-labels-idx1-ubyte" ); 

    my @label_PDL = (); # ( trainのPDL , testのPDL)で格納される想定

    while ( my $filename = shift @mnist_labelfile ) {
        push (@label_PDL , &labelload($filename));
    }

    # ラベルロードサブルーチン
    sub labelload {
        my $mnist_labels_file = shift;

        open  my $mnist_labels_fh, '<', $mnist_labels_file
          or die "Can't open file $mnist_labels_file: $!";

        # マジックナンバー
        my $labels_buffer;
        read($mnist_labels_fh, $labels_buffer, 4);
        my $magic_number = unpack('N1', $labels_buffer);
        if ($magic_number != 0x00000801) {
          die "Invalid magic number expected " . 0x00000801 . "actual $magic_number";
        }

        # ラベル数
        read($mnist_labels_fh, $labels_buffer, 4);
        my $labels_count = unpack('N1', $labels_buffer);

	undef $labels_buffer;

        my @label_x; #perl配列にPDLが入る
        my $offset = 8;
        for my $cnt ( 1 .. $labels_count ) {
            seek($mnist_labels_fh , $offset , 0 ); 
            push ( @label_x , readflex $mnist_labels_fh , [{Dims => [ 1 ] , Type => 'byte'}] ); 
            $offset++;
        }

	close $mnist_label_fh;

        my $label_2D = cat @label_x; # ( 1 , 60000)
	   $label_2D = $label_2D->transpose; # (60000 , 1);

	undef @label_x;

        #say $label_2D(0);
        
	$label_2D = convert($label_2D , double); # typeをdoubleに

        return $label_2D; 

    } # labelload

    # ( トレーニングイメージ , トレーニングラベル　, 　テストイメージ , テストラベル )
    return ( $image_PDL[0] , $label_PDL[0] , $image_PDL[1] , $label_PDL[1] );

} # mnistload

sub chg_hotone {
    my $X = shift;
    $X = topdl($X);
    # 1次元配列をhot-oneラベルに変換する
    # MNIST用
    my $T = zeros($X->nelem , 10);
    my $end = $X->nelem -1;
    for my $i ( 0 .. $end ) {
        $T($i,list($X($i))) .= 1;
    }
    return $T;
}

# 標準化、データを0-1の間に標準化する
sub normalize {
    my $X = shift;
    $X = topdl($X);

    $X /= 255.0; # 最大値で割る

    return $X;
}

1;
