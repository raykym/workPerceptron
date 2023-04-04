#!/usr/bin/env perl

# Adam

use strict;
use warnings;
use feature 'say';

# ハイパーパラメーター
my $biase = 0.14;
my $learning_rate = 0.001;
my $much_small_value = 1e-8;
my $before_moment_weight = 0.9;
my $before_velocity_weight = 0.999;

# モーメントの値
my $moment = 0;

# ヴェロシティの値
my $velocity = 0;
for (my $i = 0; $i < 10; $i++) {
  my $grad = calc_grad();
  $moment = $before_moment_weight * $moment + (1 - $before_moment_weight) * $grad;
  $velocity = $before_velocity_weight * $velocity + (1 - $before_velocity_weight) * $grad * $grad;
  
  my $cur_moment = $moment / (1 - $before_moment_weight);
  my $cur_velocity = $velocity / (1 - $before_velocity_weight);
  
  $biase -= ($learning_rate / (sqrt($cur_velocity) + $much_small_value)) * $cur_moment;

  say "grad: $grad bias: $biase";
}

# 傾きを求める
sub calc_grad {
  
  # 便宜的な値を返す
  my $grad = rand;
  
  return $grad;
}
