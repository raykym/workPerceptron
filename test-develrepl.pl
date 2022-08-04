#!/usr/bin/env perl
#
#Devel::REPL test
#
# モジュールをインタラクティブに扱って、デバッグがしやすくなる

#use strict;
#use warnings;
#use utf8;
#use feature 'say';

#binmode 'STDOUT' , ':utf8';

use Devel::REPL;

my $repl = Devel::REPL->new;

$repl->load_plugin($_) for qw( LexEnv Packages MultiLine::PPI Colors ReadLineHistory );

$repl->lexical_environment->do(<<'CODEZ');
# develop env
use FindBin;
use lib "$FindBin::Bin/../lib";
#use MyApp::Schema;
CODEZ

$repl->run;
