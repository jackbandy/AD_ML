'''test_ml_script.py: testing anomaly detection on NSL-KDD'''
'''adapted from http://karpathy.github.io/neuralnets/'''
__author__ = "Jack Bandy"
__email__ = "jaxterb@gmail.com"


import numpy as np
import csv
import collections
# Let the tensors flow
import tensorflow as tf

Dataset = collections.namedtuple('Dataset', ['data', 'target'])


def main():

    TRAINING_FILE_20P = '20 Percent Training Set.csv'
    TRAINING_FILE_SMALL = 'Small Training Set.csv'
    TRAINING_FILE_FULL = 'KDDTrain+.csv'
    TEST_FILE = 'KDDTest+.csv'

    training_set = csv_to_array(TRAINING_FILE_20P)
    test_set = csv_to_array(TRAINING_FILE_SMALL)

    # as per tensorflow's recommendation / sample code
    x_train, x_test, y_train, y_test = training_set.data, test_set.data, \
              training_set.target, test_set.target

    #Build a 3-layer DNN! (10, 20, 20 units)
    classifier = tf.contrib.learn.DNNClassifier(hidden_units=[10,20,10])
    classifier.fit(x=x_train, y=y_train, steps=200)

    accuracy_score = classifier.evaluate(x=x_test, y=y_test)["accuracy"]
    print('Accuracy: {0:f}'.format(accuracy_score))



def csv_to_array(csv_file):
    packets = csv.reader(open(csv_file), delimiter=',',dialect=csv.excel_tab)

    protocol_map = {}
    service_map = {}
    flag_map = {}
    label_map = {}

    labels = []
    features = []

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
        labels.append(np.int(generate_number(tmp.pop(),label_map)))
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




if __name__ == '__main__':
    main()


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



def label_groups():
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
					'guess_password':'r2l',
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

