'''test_ml_script.py: testng anomaly detection on NSL-KDD'''
'''adapted from http://karpathy.github.io/neuralnets/'''
__author__ = "Jack Bandy"
__email__ = "jaxterb@gmail.com"


import numpy as np
import csv
import collections
import time
# Let the tensors flow
import tensorflow as tf

Dataset = collections.namedtuple('Dataset', ['data', 'target'])
label_numbers = {'normal':0,'dos':1,'probe':2,'u2r':3,'r2l':4}

def main():

    TRAINING_FILE_20P = '20 Percent Training Set.csv'
    TRAINING_FILE_SMALL = 'Small Training Set.csv'
    TRAINING_FILE_FULL = 'KDDTrain+.txt'
    TEST_FILE = 'KDDTest+.txt'

    the_training_set = csv_to_array(TRAINING_FILE_FULL)
    the_test_set = csv_to_array(TRAINING_FILE_SMALL)

    unit_trials = []
    unit_trials.append([10,20,10])
    unit_trials.append([100,200,100])

    for trial in unit_trials:
        run_dnn_with_units(the_training_set,the_test_set,trial)



def run_dnn_with_units(training_set,test_set,units_array):
    # as per tensorflow's recommendation / sample code
    x_train, x_test, y_train, y_test = training_set.data, test_set.data, \
              training_set.target, test_set.target


    #Build a 3-layer DNN! (10, 20, 20 units)
    start = time.clock()
    classifier = tf.contrib.learn.DNNClassifier(hidden_units=units_array)
    classifier.fit(x=x_train, y=y_train, steps=200)
    stop = time.clock()
    print('-------------------------')
    print('DNN with hidden units: ' + str(units_array))
    print('Seconds elapsed: {}'.format(stop - start))

    accuracy_stuff = classifier.evaluate(x=x_test, y=y_test)
    print('Accuracy: {0:f}'.format(accuracy_stuff['accuracy']))
    print('Other stuff: ' + str(accuracy_stuff))

    print('-------------------------')


def csv_to_array(csv_file):
    packets = csv.reader(open(csv_file), delimiter=',',dialect=csv.excel_tab)

    protocol_map = {}
    service_map = {}
    flag_map = {}

    labels = []
    features = []

    label_groups = get_label_groups()

    for packet in packets:
        tmp = []
        for feature_index in range(0,len(packet)):
            if feature_index is 1:
                # Handle protocol numbering
                tmp.append(np.float32(generate_number(packet[1],protocol_map)))
            elif feature_index is 2:
                # Handle service numbering
                tmp.append(np.float32(generate_number(packet[2],service_map)))
            elif feature_index is 3:
                # Handle flag numbering
                tmp.append(np.float32(generate_number(packet[3],flag_map)))
            else:
                tmp.append(packet[feature_index])
        # Last item in the list is unneeded
        del tmp[len(tmp)-1]
        # Second-to-last-item is the label
        label = tmp.pop()
        # Figure out which group it falls into
        if not label == 'normal':
            label = label_groups[label]
        label_number = label_numbers[label]
        if label_number != 0: label_number = 1
        labels.append(np.int(label_number))
        features.append(np.array(tmp))

    labels = np.array(labels)
    features = np.array(features)
    return (Dataset(features,labels))



def generate_number(the_str,the_map):
    # Create numeric labels for strings using a given map
    # i.e. the_map = {'tcp':0,'ftp':1,etc}

    if not the_str in the_map:
        the_map[the_str] = len(the_map.keys())
    return the_map[the_str]




def feature_names(): 
    return [
        # basic features (0-8)
        'duration',
        'protocol_type',
        'service',
        'flag',
        'src_bytes',
        'dst_bytes',
        'land',
        'wrong_fragment',
        'urgent',
        # content related features (9-21)
        'hot',
        'num_failed_logins',
        'logged_in',
        'num_compromised',
        'root_shell',
        'su_attempted',
        'num_root',
        'num_file_creations',
        'num_shells',
        'num_access_files',
        'num_outbound_cmds',
        'is_hot_login',
        'is_guest_login',
        # time related attributes (22-30)
        'count',
        'srv_count',
        'serror_rate',
        'srv_serror_rate',
        'rerror_rate',
        'srv_rerror_rate',
        'same_srv_rate',
        'diff_srv_rate',
        'srv_diff_host_rate',
        # host based traffic features (31-40)
        'dst_host_count',
        'dst_host_srv_count',
        'dst_host_same_srv_rate',
        'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate',
        'dst_host_serror_rate',
        'dst_host_srv_serror_rate',
        'dst_host_rerror_rate',
        'dst_host_rerror_rate',
        'dst_host_srv_rerror_rate'
    ]



def get_label_groups():
    return  {	    # denial of service attacks
					'back':'dos',
					'land':'dos',
					'neptune':'dos',
					'pod':'dos',
					'smurf':'dos',
					'teardrop':'dos',
					'apache2':'dos',
					'udpstorm':'dos',
					'processtable':'dos',
					'worm':'dos',
					# probe attacks
					'satan':'probe',
					'ipsweep':'probe',
					'nmap':'probe',
					'portsweep':'probe',
					'mscan':'probe',
					'saint':'probe',
					# root to local (r2l) attacks
					'guess_passwd':'r2l',
					'ftp_write':'r2l',
					'imap':'r2l',
					'phf':'r2l',
					'multihop':'r2l',
					'warezmaster':'r2l',
					'warezclient':'r2l' ,
					'spy':'r2l',
					'xlock':'r2l',
					'xsnoop':'r2l',
					'snmpguess':'r2l',
					'snmpgetattack':'r2l',
					'httptunnel':'r2l',
					'sendmail':'r2l',
					'named':'r2l',
					# user to root (u2r) attacks
					'buffer_overflow':'u2r',
					'loadmodule':'u2r',
					'rootkit':'u2r',
                    'perl':'u2r',
                    'sqlattack':'u2r',
					'xterm':'u2r',
					'ps':'u2r'
			}




if __name__ == '__main__':
    main()
