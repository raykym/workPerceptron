#!/usr/bin/env perl
#
## Tie::IxHashでハッシュを宣言すると、書き込み順序を記録したハッシュを使うことができる。
# 明示的にリファレンスを指定すれば、リファレンス指定も可能になる。
# existsも使えるので、判定にも使える。
# OOの使い方は今の所便利さを思いつかない

use strict;
use warnings;
use utf8;

binmode 'STDOUT' , ':utf8';

use feature 'say';
use Tie::IxHash;

tie( my %hash_seq , 'Tie::IxHash') ;
my $hash_ref = \%hash_seq;

 # 順番が記録されている
 $hash_seq{three} = 3;
 $hash_seq{one} = 1;
 $hash_seq{two} = 2;

 # リファレンスで追加
 $hash_ref->{four} = 4;

   for my $key ( keys %hash_seq ) {
       say $hash_seq{$key};
   }
   say "";

   # 書き換え オブジェクトのサブルーチンではなく、ハッシュ操作を前提として書き換えが行われている。 
   $hash_ref->{one} = 11;

   for my $key ( keys %hash_seq ) {
       say $hash_seq{$key};
   }

   say "revers";
   # 3-> 11 -> 2 -> 4 昇順 を逆順にする。
   my @keylist;
   for my $key ( keys %hash_seq ) {
       push(@keylist , $key);
   }

   while ( my $key = pop @keylist ) {
	   #say $hash_seq{$key}; 
      say $hash_ref->{$key}; 
   }


#my $hash_seq = Tie::IxHash->new( one => 1 , two => 2 , three => 3);
#while ( my $key = $hash_seq->Keys() ) {
#    say $hash_seq->{$key};
#}
