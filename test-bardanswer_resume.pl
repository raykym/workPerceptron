#!/usr/bin/env perl

# bardに割り込みで中断から復帰する処理のサンプル。。。これは足りない・・・

use utf8;
$|=1;



sub interrupt_handler {
  print "interrupt occurred\n";
}

sub main {
  while (1) {
    print "processing\n";
    sleep 1;
  }
}

$pid = fork();
if ($pid) {
  # parent process
  wait;
  print "resuming\n";
  main();
} else {
  # child process
  signal SIGINT, \&interrupt_handler;
  main();
}
