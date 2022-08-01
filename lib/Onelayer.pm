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

