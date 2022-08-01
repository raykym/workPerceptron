#!/usr/bin/env perl
#
# simple perceptronの学習ななど、自作実装で動作を確認する。
# AND OR NAND NORについて動作することを確認
#
use strict;
use warnings;
use utf8;
use feature ':5.13';

binmode 'STDOUT' , ':utf8';

use Time::HiRes qw/ time /;
use Data::Dumper;
#use Devel::Size;
#use Devel::Cycle;
#use Devel::Peek;
use Data::Dumper;

use lib './lib';
use Perceptron;
use Multilayer;

#use Scalar::Util qw/ weaken /;


$|=1;

srand();

sub Logging {                                                                                                                                   
        my $logline = shift;                                                                                                                    
        my $dt = time();                                                                                                                        
        say "$dt | $logline";                                                                                                                   
                                                                                                                                                
        undef $dt;                                                                                                                              
        undef $logline;                                                                                                                         
                                                                                                                                                
        return;                                                                                                                                 
}                                                                                                                                               
   
my $learndata_XORgate = [
                      { 
                        class => 0 ,                                                                                                           
                        input => [ 1 , 1 ]                                                                                                      
                      },                                                                                                                        
                      {                                                                                                                         
                        class => 1 ,                                                                                                            
                        input => [ 0 , 1 ]                                                                                                     
                      },                                                                                                                        
                      {                                                                                                                         
                        class => 1 ,                                                                                                            
                        input => [ 1 , 0 ]                                                                                                     
                      },                                                                                                                        
                      {                                                                                                                         
                        class => 0 ,                                                                                                           
                        input => [ 0 , 0 ]  
                      },
                        ];

# buffer layer

	#my $buff_layer = [];
	#push(@{$buff_layer} , Perceptron->new);
	#push(@{$buff_layer} , Perceptron->new);
    my $buff_layer = Onelayer->new;
    my $structure = { node_count => 1 ,
	              waits_count => 1,
	            };
    $buff_layer->node_init($structure);

	# x1000 buffer
	#my $wa = $buff_layer->[0]->waits([ 1000 , 0 ]);
	#my $ba = $buff_layer->[0]->bias(1);
	#my $wb = $buff_layer->[1]->waits([ 0 , 1000 ]);
	#my $bb = $buff_layer->[1]->bias(1);

    my $waits_list = [
                         { 
			   node_num => 0 ,
			   waits => [ 1000 , 0 ],
		         },
			 { 
			   node_num => 1 ,
			   waits => [ 0 , 1000 ],
		         }
	             ];

    for my $w ( @{$waits_list} ) {
        $buff_layer->node_waits($w);
    }

    for (my $n=0; $n<=$structure->{node_count}; $n++) {
        $buff_layer->node_bias($n , 1);
    }


=pod
	my @out = ();
	for my $sample ( @{$learndata_XORgate} ) {
	    $buff_layer->input($sample->{input});
	    my $res = $buff_layer->nodes_calcReLU();
	    push ( @out , $res );
	}
	Logging("@{$_->{input}}")  for @{$learndata_XORgate};
	say "";
	for my $out (@out) {
	    Logging("@{$out}");
	}
=cut

    my $structure_multi = {
                      layer_member  => [ 1 , 0 ],
                      input_count => 1 ,
                      learn_rate => 0.034
                    };

    my $multilayer = Multilayer->new();
       $multilayer->layer_init($structure_multi);

     my $newdata =[];
    for my $sample ( @{$learndata_XORgate} ) {
        $buff_layer->input($sample->{input});
        my $out = $buff_layer->nodes_calcReLU();
	push (@{$newdata} , { class => $sample->{class} , input => $out } ); 
    }    

    my $learn_flg = 1;
    my $learn_count = 0;
    while ( $learn_flg ) {
	    $learn_count++;
	    Logging("learn count: $learn_count ");
	    if ( $learn_count >= 2000 ) {
		$learn_flg = 0;
                Logging("DEBUG: learn_count over 2000");
		exit;
	    }

        my $stat = $multilayer->learn($newdata);

	if ($stat eq 'learned' ) {
            $learn_flg = 0; 
	}
    }
	Logging("Finish!! ");

        $multilayer->dump_structure();





package Onelayer;

use Carp;
use Clone;
use feature 'say';


sub new {
    my $proto = shift;
    my $class = ref $proto || $proto;

    my $self = {};
       $self->{nodes} = [];
       $self->{input} = undef;

    bless $self , $class;

    return $self;
}

sub node_init {
    my $self = shift;
=pod
    # node count  ARG#
    # waits count (befor nodes number) ARG#
     { node_count => XXX,
       waits_count => yyy,
     } 
=cut
    if (@_) {
        if ($_[0] =~ /HASH/ ) {
            $self->{node_count} = $_[0]->{node_count}; 
            $self->{waits_count} = $_[0]->{waits_count};
	} else {
            croak "params miss match...";
	    exit;
	}
    } else {
        croak "no params....";
	exit;
    }

    for my $node ( 0 .. $self->{node_count} ) {
	push( @{$self->{nodes}} , Perceptron->new);
        $self->{nodes}->[$node]->waitsinit($self->{waits_count} , $self->{node_count});
    }

}

sub node_waits {
    my $self = shift;

    # each node set waits
=pod
    {  node_num => 'ARG#' ,
       waits => [ xx , xx , xxx ],
    }
=cut
    my $node_num;
    my $waits = undef;
    if (@_) {
        if ($_[0] =~ /HASH/ ) {
            $node_num = $_[0]->{node_num}; 
            $waits = $_[0]->{waits};
	} else {
            croak "input miss match...";
	}
    } else {
        croak "no ARG....";
    }

    my $res = $self->{nodes}->[$node_num]->waits($waits);

    return $res;
}

sub node_bias {
    my $self = shift;
    # node num & bias value
    my $node = undef;
    my $value = undef;
    if (@_) {
        if ( $_[0] =~ /ARRAY/ || $_[0] =~ /HASH/ ) {
            croak "ARG is reffernce ....";
	} elsif ( defined $_[1] ) {
            $node = $_[0];
	    $value = $_[1];
	} else {
            $node = $_[0];
	}	
    } else {
        croak "no ARG ....";
    }

    my $bias = undef;
    if ( defined $value ) {
        $bias = $self->{nodes}->[$node]->bias($value);
    } else {
        $bias = $self->{nodes}->[$node]->bias();
    }

    return $bias;
}

sub input {
    my $self = shift;
    # input setup only...
    if (@_) {
        if ( $_[0] =~ /ARRAY/ ) {
            $self->{input} = $_[0];
	} else {
            croak "input miss match...";
	}
    } else {
        croak "no input missing ...";
    }
}

sub nodes_calcReLU {
    my $self = shift;

    my $out = [];
    for my $node ( @{$self->{nodes}} ) {
        $node->input($self->{input});
	my $res = $node->calcReLU();
	push (@{$out} , $res );
    }

    return $out;
}

