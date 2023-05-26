package Adam_optimizer;

use v5.32;
use utf8;

use PDL;
use PDL::Core ':Internal';
use PDL::NiceSlice;

binmode 'STDOUT' , ':utf8';

sub new {
    my $proto = shift;
    my $class = ref $proto || $proto;

    my $self = {};
       $self->{lr} = 0.001;
       $self->{beta1} = 0.9;
       $self->{beta2} = 0.999;
       $self->{itre} = 0;
       $self->{m} = undef;
       $self->{v} = undef;

    bless $self , $class;

    return $self;
}

sub update {
    my ($self , $params , $grads ) = @_;
    # $paramsは$network->{params}が入る

    if ( ! defined $self->{m} ) {
        $self->{m} = {};
	$self->{v} = {};
	for my $key ( keys %{$params} ) {
            $self->{m}->{$key} = zeros($params->{$key});
	    $self->{v}->{$key} = zeros($params->{$key});
	}
    } # if

    $self->{itre}++;

    my $lr_t = $self->{lr} * sqrt ( 1.0 - $self->{beta2} ** $self->{itre} ) / (1.0 - $self->{beta1} ** $self->{itre});

    #  say "lr_t: $lr_t";

    for my $key (keys %{$params}) {
        $self->{m}->{$key} += (1 - $self->{beta1}) * ($grads->{$key} - $self->{m}->{$key});
	$self->{v}->{$key} += (1 - $self->{beta2}) * ($grads->{$key} ** 2 - $self->{v}->{$key});

        $params->{$key} -= $lr_t * $self->{m}->{$key} / (sqrt ($self->{v}->{$key}) + 0.0000001);

    } # for key 

} # update



1;
