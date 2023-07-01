package Util;

# ゼロから作るdeep learning2のサブルーチン集

use utf8;
binmode 'STDOUT' , ':utf8';

use PDL;
use PDL::Core ':Internal';
use PDL::NiceSlice;

#英文入力を想定
sub preprocess {
    my $text = shift;
       $test = lc($text); #小文字に揃える
       $text =~ s/./ ./g; #ピリオド前に空白
    my @word = split(" " , $text); #空白で分割

    my $word_to_id = {};
    my $id_to_word = {};
    for my $w (@word) {
        if ( ! exists($word_to_id->{$w}) ) {
            my $new_id = keys %{$word_to_id};
               $word_to_id->{$w} = $new_id;
               $id_to_word->{$new_id} = $w; 
        } 
    my $corpus = pdl(sort keys %{$id_to_word}); #これでできるか不明
    }
    return ($corpus , $word_to_id , $id_to_word);
}











1;
