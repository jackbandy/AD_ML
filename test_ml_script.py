'''test_ml_script.py: testng anomaly detection on NSL-KDD'''
'''adapted from http://karpathy.github.io/neuralnets/'''
__author__ = "Jack Bandy"
__email__ = "jaxterb@gmail.com"

#TODO: make categorical labels into binary labels (ex. 'tcp' -> is_tcp = 1)


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


    training_file = np.array(simple_csv_to_array(TRAINING_FILE_FULL))
    testing_file = np.array(simple_csv_to_array(TEST_FILE))
    columns = len(training_file[0])
    # Delete the last column
    training_file = np.delete(training_file,columns-1,1)
    testing_file = np.delete(testing_file,columns-1,1)

    training_labels = training_file[:,columns-2]
    testing_labels = testing_file[:,columns-2]
    training_features = np.delete(training_file,columns-2,1)
    testing_features = np.delete(testing_file,columns-2,1)


    feature_lookups = {}
    for i in range(0,len(training_file[0])):
        try:
            np.float32(training_file[0][i])
        except ValueError as e:
            feature_lookups[i] = map_for_labels(training_file[:,i])
            # detected non-numeric value, make a value->number map
    print(str(feature_lookups))

    label_lookups = get_label_groups()


    training_features = binarize_features(training_file,feature_lookups)
    test_features = binarize_features(testing_file,feature_lookups)
    training_labels = binarize_labels(training_labels,label_lookups)
    test_labels = binarize_labels(testing_labels,label_lookups)

    the_training_set = Dataset(training_features,training_labels)
    the_test_set = Dataset(test_features,test_labels)

    unit_trials = []
    unit_trials.append([41])
    unit_trials.append([41,40,40])
    unit_trials.append([41,20,20,20])
    unit_trials.append([41,25,20])

    step_trials = []
    step_trials.append(200)
    step_trials.append(300)
    step_trials.append(400)

    for trial in unit_trials:
        for num_steps in step_trials:
            run_dnn_with_units_steps(the_training_set,the_test_set,trial,num_steps)





def run_dnn_with_units_steps(training_set,test_set,units_array,num_steps):
    # as per tensorflow's recommendation / sample code
    x_train, x_test, y_train, y_test = training_set.data, test_set.data, \
              training_set.target, test_set.target

    #Build a DNN!
    start = time.clock()
    classifier = tf.contrib.learn.DNNClassifier(hidden_units=units_array)
    classifier.fit(x=x_train, y=y_train, steps=num_steps)
    stop = time.clock()
    print('-------------------------------------')
    print('DNN with hidden units: ' + str(units_array))
    print('Number of steps: ' + str(num_steps))
    print('Seconds elapsed: {}'.format(stop - start))
    accuracy_stuff = classifier.evaluate(x=x_test, y=y_test)
    print('Accuracy: {0:f}'.format(accuracy_stuff['accuracy']))
    print('Other stuff: ' + str(accuracy_stuff))

    print('-------------------------------------')




def simple_csv_to_array(csv_file):
    to_return = []
    packets = csv.reader(open(csv_file), delimiter=',',dialect=csv.excel_tab)
    for packet in packets:
        tmp = []
        for feature_index in range(0,len(packet)):
            tmp.append(packet[feature_index])
        to_return.append(tmp)
    return to_return



def binarize_labels(raw_labels,label_lookups):
    labels = []
    # Figure out which group it falls into
    for label in raw_labels:
        if not label == 'normal':
            label = label_lookups[label]
        label_number = label_numbers[label]
        # Make all non-normal packets anomalies
        # Just for now :)
        if label_number != 0: label_number = 1
        labels.append(np.int(label_number))

    return np.array(labels)
    



def binarize_features(raw_array,label_lookups):
    features = []

    for packet in raw_array:
        tmp = []
        for feature_index in range(0,len(packet)):
            if feature_index in label_lookups.keys():
                binarize = [np.float32(0.0)]*len(label_lookups[feature_index])
                label = packet[feature_index]
                if(label_lookups[feature_index].get(label)):
                    binarize[label_lookups[feature_index][label]] = np.float32(1.0)
                tmp.extend(binarize)
            else:
                tmp.append(np.float32(packet[feature_index]))

        features.append(np.array(tmp))


    features = np.array(features)
    return features
    




def kdd_csv_to_array(csv_file):
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

        # Make all non-normal packets anomalies
        if label_number != 0: label_number = 1

        labels.append(np.int(label_number))
        features.append(np.array(tmp))

    labels = np.array(labels)
    features = np.array(features)
    print("Protocols:")
    print(str(protocol_map))
    print("Services:")
    print(str(service_map))
    print("Flags:")
    print(str(flag_map))
    return (Dataset(features,labels))




def map_for_labels(labels):
    # given a list of values, return a map of label->val
    # i.e. {'tcp':0,'ftp':1,etc}
    to_return = {}
    for label in labels:
        generate_number(label,to_return)
    return to_return



def generate_number(the_str,the_map):
    # Create numeric labels for strings using a given map
    # i.e. the_map = {'tcp':0,'ftp':1,etc}
    #      the_str = 'ftp'
    #      returns 1 

    if not the_str in the_map:
        the_map[the_str] = len(the_map.keys())
    return the_map[the_str]




def feature_names(): 
    return [
        # c = continuous feature
        # d = discrete feature
        # basic features (0-8)
        {'name':'duration','type':'c'},
        {'name':'protocol_type','type':'d'},
        {'name':'service','type':'d'},
        {'name':'flag','type':'d'},
        {'name':'src_bytes','type':'c'},
        {'name':'dst_bytes','type':'c'},
        {'name':'land','type':'d'},
        {'name':'wrong_fragment','type':'c'},
        {'name':'urgent','type':'c'},
        # content related features (9-21)
        {'name':'hot','type':'c'},
        {'name':'num_failed_logins','type':'c'},
        {'name':'logged_in','type':'d'},
        {'name':'num_compromised','type':'c'},
        {'name':'root_shell','type':'d'},
        {'name':'su_attempted','type':'d'},
        {'name':'num_root','type':'c'},
        {'name':'num_file_creations','type':'c'},
        {'name':'num_shells','type':'c'},
        {'name':'num_access_files','type':'c'},
        {'name':'num_outbound_cmds','type':'c'},
        {'name':'is_hot_login','type':'d'},
        {'name':'is_guest_login','type':'d'},
        # time related attributes (22-30)
        {'name':'count','type':'d'},
        {'name':'srv_count','type':'d'},
        {'name':'serror_rate','type':'c'},
        {'name':'srv_serror_rate','type':'c'},
        {'name':'rerror_rate','type':'c'},
        {'name':'srv_rerror_rate','type':'c'},
        {'name':'same_srv_rate','type':'c'},
        {'name':'diff_srv_rate','type':'c'},
        {'name':'srv_diff_host_rate','type':'c'},
        # host based traffic features (31-40)
        {'name':'dst_host_count','type':'d'},
        {'name':'dst_host_srv_count','type':'d'},
        {'name':'dst_host_same_srv_rate','type':'c'},
        {'name':'dst_host_diff_srv_rate','type':'c'},
        {'name':'dst_host_same_src_port_rate','type':'c'},
        {'name':'dst_host_srv_diff_host_rate','type':'c'},
        {'name':'dst_host_serror_rate','type':'c'},
        {'name':'dst_host_srv_serror_rate','type':'c'},
        {'name':'dst_host_rerror_rate','type':'c'},
        {'name':'dst_host_rerror_rate','type':'c'},
        {'name':'dst_host_srv_rerror_rate','type':'c'}
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
                    'mailbomb':'dos',

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
