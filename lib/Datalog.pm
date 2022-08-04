package Datalog;

use Carp;

use DBI;
use Data::Dumper;
use Time::HiRes qw/ time /;

# epocタイムでテーブルを作成して、waitsデータを記録していく
# スクリプト1回でテーブル1つの想定
# sqlite3なのでファイルを削除して、新規にログを取るケースも在ると

sub new {
    my $proto = shift;
    my $class = ref $proto || $proto;

    my $self = {};
    if (@_) {
        if ($_[0] ne "" ) {
            $self->{filename} = $_[0] || 'multilayer' ;
	} else {
            croak "file name error";
	}
    }

       $self->{filename} ||= 'multilayer' ;
       $self->{table_name} = "";
       $self->{dbh} = "";

    bless $self , $class;

    $self->_init();

    return $self;
}


sub _init {
    my $self = shift;
    # sqlite3のデータベース、テーブルを準備

    my $dt = time();
       $dt =~ tr /./_/;
       $self->{table_name} = "table_$dt";    

       $self->{dbh} = DBI->connect("dbi:SQLite:dbname=$self->{filename}.sqlite3" , undef , undef , { 
		           AutoCommit => 1,
			   RaiseError => 1,
			   sqlite_see_if_its_a_number => 1
		   });

       $self->{dbh}->do("create table $self->{table_name} (id integer primary key autoincrement , data text)");

}

sub addlog {
    my $self = shift;
=pod
    # ARGV layer data dump
    $array ={ 
              waits => $new_layerwaits,
              bias => $new_layerbias,
            },
             
=cut

    if (@_) {
        if (defined $_[0]) {
            my $data_strings = $_[0];

            my $insert_sth = $self->{dbh}->prepare("insert into $self->{table_name} (data) values (?)");
               $insert_sth->execute($data_strings);        

        } else {
            croak "input data error!";
        }
    } else {
        croak "no param Error!";
    }
}

sub begin_work {
    my $self = shift;

    $self->{dbh}->begin_work;
}

sub commit {
    my $self = shift;
 
   $self->{dbh}->commit;
}

sub finish_rollback {
    my $self = shift;

    $self->{dbh}->finish;
    $self->{dbh}->rollback;
}

sub DESTROY {
    my $self = shift;

    #  $self->{dbh}->commit();
}


1;
