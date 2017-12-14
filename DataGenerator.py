import numpy as np
import json
from glob import iglob
from pymongo.cursor import Cursor
from io import IOBase

# DataGenerator class
class DataGenerator(object):
    'Generates data for Keras'
    def __init__(self, file_pattern, input_shape, fields, batch_size = 32, n_classes = None, shuffle = True, seed = 0):
        'Constructor'
        np.random.seed(seed)
        self.file_pattern = file_pattern
        self.input_shape = input_shape
        self.fields = fields
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle

    def generate(self, id_list):
        'Generates batches of samples'
        # Infinite loop
        while True:
            # Generate order of exploration of dataset
            indexes = self.__get_exploration_order(id_list)

            # Generate batches
            imax = int(len(indexes) / self.batch_size)
            for i in range(imax):
                # Find list of IDs
                id_list_batch = [id_list[k] for k in indexes[i * self.batch_size:(i + 1) * self.batch_size]]

                # Generate data
                X, y = self.__data_generation(id_list_batch)

                yield X, y

    def __get_exploration_order(self, id_list):
        'Generates order of exploration'
        # Find exploration order
        indexes = np.arange(len(id_list))
        if self.shuffle == True:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, id_list_batch):
        'Generates data of batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.input_shape))
        y = np.empty((self.batch_size))

        # Generate data
        line_counter = 0
        match_counter = 0
        while match_counter < self.batch_size:
            for file_name in iglob(self.file_pattern):
                with open(file_name, 'r') as file_stream:
                    for line in file_stream:
                        if line_counter in id_list_batch:
                            doc = json.loads(line.rstrip("\n").replace("'","\""))
                            # Store input
                            X_formatted = augment_features(X_format(doc[self.fields[0]]))
                            X[match_counter] = np.pad(X_formatted, [(0, self.input_shape[i] - X_formatted.shape[i]) for i in range(len(self.input_shape))], 'constant')
                            # Store target
                            y[match_counter] = y_format(doc[self.fields[1]])
                            # Augment
                            X[match_counter] = self.__augment_data(X[match_counter])
                            match_counter += 1
                            if match_counter == self.batch_size:
                                break
                        line_counter += 1

        return X, self.__binarize(y)

    def __binarize(self, y):
        'Returns labels in binary NumPy array'
        if self.n_classes != None:
            return np.array([[1 if y[i] == j else 0 for j in range(self.n_classes)]
                             for i in range(y.shape[0])])
        else:
            return y

    def __augment_data(self, X):
        'Augment the data by adding a random transformation on the features'
        return X
        #return X[np.random.permutation(X.shape[0]),:]

def augment_features(X):
    'Augment the raw data features'
    #return np.append(X, X ** 2)
    return X

def X_format(X):
    'Change the format of an input value'
    return np.array(eval(str(X).replace(" ","").replace("{","[").replace("}","]")), dtype = int)

def y_format(y):
    'Change the format of an target value'
    return np.array(y, dtype = int)