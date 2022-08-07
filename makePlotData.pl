#!/usr/bin/env perl
#
# make GnuPlot data from datastore.
# each nodes waits inclination on calc step
#
# need multilayer.sqlite3 file default
#

use Devel::REPL;

my $repl = Devel::REPL->new;

$repl->load_plugin($_) for qw( LexEnv Packages MultiLine::PPI Colors ReadLineHistory );

$repl->lexical_environment->do(<<'CODEZ');
# develop env
use strict;
use warnings;
use utf8;
binmode 'STDOUT' , ':utf8';
use feature 'say';
my $db = Dataplot->new;

# if you need dbname change!!!
# $db->dbname('XXXXXXX');

   $db->init_mod;
my @tables = $db->show_tables;
say for @tables;
say "table name over";
CODEZ
#######################################

print << 'EOF';
# follow commands

my $res = $db->load_table('table_NAME');

say "plot tern [ 0 - $res ]";

$db->plotdata(0);
EOF

$repl->run();



package Dataplot;
use DBI;
use Carp;
use List::Util;
use Data::Dumper;

sub new {
    my $proto = shift;
    my $class = ref $proto || $proto;

    my $self = {};
       $self->{dbname} = undef;
       $self->{dbh} = undef;
       $self->{tables} = undef; # no use
       $self->{tabledata} = [];
       $self->{list_count} = undef;
       
    bless $self , $class;

    return $self;
}

sub dbname {
    my $self = shift;

    if (@_) {
        if ( $_[0] eq "" ) {
            croak "dbname error!";
        } else {
            $self->{dbname} = $_[0];
        }
    }

    return $self->{dbname};
}

sub init_mod {
    my $self = shift;

    if ( defined $self->{dbname} ) {
        $self->{dbh} = DBI->connect("dbi:SQLite:dbname=$self->{dbname}.sqlite3",undef,undef,{
                        AutoCommit => 1,
                        RaiseError => 1,
                        sqlite_see_if_its_a_number => 1
                    });
    } else {
        $self->{dbh} = DBI->connect("dbi:SQLite:dbname=multilayer.sqlite3",undef,undef,{
                        AutoCommit => 1,
                        RaiseError => 1,
                        sqlite_see_if_its_a_number => 1
                    });
    }
}

sub show_tables {
    my $self = shift;

    my @table_list = $self->{dbh}->tables();

    return @table_list;
}

sub load_table {
    my $self = shift;
    # a table load to object
    # return structure data count

    my $sth = undef;
    my $res_arrayref = undef;
    # argv table name
    if (@_) {
        if ( $_[0] eq "" ) {
            croak "no table name";
	} else {
            $sth = $self->{dbh}->prepare("select data from $_[0]"); 
	    $sth->execute();
            $res_arrayref = $sth->fetchall_arrayref();  # arrayref in text dump
	}
    } else {
        croak "no table name ";
    }

    for my $ref (@{$res_arrayref}) {
        for my $text (@{$ref}) {
            my $VAR1;    # DumperがVAR1で定義しているので、事前に用意しておく
            eval $text;
            push( @{$self->{tabledata}} , $VAR1 );
        }
    }

    my @table = @{$self->{tabledata}};
       $self->{list_count} = $#table;  # count of structure data

    undef @table;
    
    return $self->{list_count};
}

sub plotdata {
    my $self = shift;
    # set count of structure data number

    my $set_count = undef;
    if (@_) {
        if ( $_[0] =~ /[0-9]+/ ) {
            $set_count = $_[0];
	} else {
            croak "input error";
	}
    } else {
        croak "no number...";
    }

    my @tabledata = @{$self->{tabledata}}; # table all data
    
    my $pice_data = $self->{tabledata}->[$set_count];
    
    #print Dumper $pice_data;

    my @tmp = @{$pice_data->{layer_init}->{layer_member}};
    my $layer_count = $#tmp;
    undef @tmp;

    #open (my $gp , '|-' , 'gnuplot') or die 'no gnuplot';
    open (my $gp , '|-' , 'gnuplot -persist') or die 'no gnuplot'; # オプションを付けないとグラフが消える

    #say $gp "set xrange [ 500 : 500] ";
    #say $gp "set yrange [ 500 : 500 ]";

    my @pfunc;
    for my $l ( 0 .. $layer_count ) {
        for my $n ( 0 .. $pice_data->{layer_init}->{layer_member}->[$l] ) {
            # 通常はARRAYデータ
            if ( $pice_data->{waits} =~ /ARRAY/ ) {
                my $wait_sum = List::Util::sum @{$pice_data->{waits}->[$l]->[$n]};
                my $bias = $pice_data->{bias}->[$l]->[$n];
           #    print "l: $l n: $n waits_sum: $wait_sum \n";
	        push (@pfunc , "$wait_sum * x + $bias");  # ノード毎に傾きの式を記録する
	    } elsif ($pice_data->{waits} =~ /HASH/ ) {
		# HASHデータの場合 
		my $wait_sum = List::Util::sum @{$pice_data->{waits}->{$l}->{$n}};
                my $bias = $pice_data->{bias}->{$l}->{$n};
	        push (@pfunc , "$wait_sum * x + $bias");  # ノード毎に傾きの式を記録する
           } 
	} # for $n
    } # for $l

    my $func_strings = join ("," , @pfunc ); # カンマ区切りで式を記述する
    
    say $gp "plot $func_strings";

    close $gp;

    undef @tabledata;
    undef @pfunc;
    undef $func_strings;

}

sub dumpdata {
    my $self = shift;
    # set count of structure data number

    my $set_count = undef;
    if (@_) {
        if ( $_[0] =~ /[0-9]+/ ) {
            $set_count = $_[0];
	} else {
            croak "input error";
	}
    } else {
        croak "no number...";
    }

    my @tabledata = @{$self->{tabledata}}; # table all data
    
    my $pice_data = $self->{tabledata}->[$set_count];
    
    print Dumper $pice_data;
}
