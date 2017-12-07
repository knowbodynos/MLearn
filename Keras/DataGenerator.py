import numpy as np
import json
from pymongo.cursor import Cursor
from io import IOBase

# DataGenerator class
class DataGenerator(object):
    'Generates data for Keras'
    def __init__(self, json_generator, input_shape, x_field, y_field, batch_size = 32, n_classes = None, x_dtype = int, y_dtype = int, x_format = lambda x: x, y_format = lambda y: y, augment = lambda x: x, shuffle = True):
        'Constructor'
        assert isinstance(json_generator, IOBase) or isinstance(json_generator, Cursor)
        self.json_generator = json_generator
        self.input_shape = input_shape
        self.x_field = x_field
        self.y_field = y_field
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.x_dtype = x_dtype
        self.y_dtype = y_dtype
        self.x_format = x_format
        self.y_format = y_format
        self.augment = augment
        self.shuffle = shuffle

    #def __del__(self):
    #    'Destructor'
    #    self.json_generator.close()

    def generate(self, id_list):
        'Generates batches of samples'
        # Infinite loop
        while True:
            if isinstance(self.json_generator, IOBase):
                # Generate order of exploration of dataset
                indexes = self.__get_exploration_order(id_list)

                # Generate batches
                imax = int(len(indexes)/self.batch_size)
                for i in range(imax):
                    # Find list of IDs
                    id_list_batch = [id_list[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]

                    # Generate data
                    X, y = self.__data_generation(id_list_batch)

                    yield X, y
            elif isinstance(self.json_generator, Cursor):
                # Generate batches
                imax = int(len(id_list)/self.batch_size)
                for i in range(imax):
                    # Find list of IDs
                    id_list_batch_ordered = id_list[i*self.batch_size:(i+1)*self.batch_size]
                    # Generate order of exploration of dataset
                    indexes = self.__get_exploration_order(id_list_batch_ordered)
                    id_list_batch = [id_list_batch_ordered[k] for k in indexes]

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

    '''
    def __file_data_generation(self, id_list_batch):
        'Generates data of batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.input_shape), dtype = self.x_dtype)
        y = np.empty((self.batch_size), dtype = self.y_dtype)

        # Generate data
        line_counter = 0
        match_counter = 0
        for line in self.json_generator:
            if line_counter in id_list_batch:
                doc = json.loads(line.rstrip("\n").replace("'","\""))
                # Store input
                X[match_counter] = self.augment(np.array(self.x_format(doc[self.x_field]), dtype = self.x_dtype)).reshape((*self.input_shape))
                # Store target
                y[match_counter] = np.array(self.y_format(doc[self.y_field]), dtype = self.y_dtype)
                match_counter += 1
                if match_counter == self.batch_size:
                    break
            line_counter += 1

        return X, self.__binarize(y)
    '''

    def __data_generation(self, id_list_batch):
        'Generates data of batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.input_shape), dtype = self.x_dtype)
        y = np.empty((self.batch_size), dtype = self.y_dtype)

        # Generate data
        doc_counter = 0
        match_counter = 0
        for doc in self.json_generator:
            if doc_counter in id_list_batch:
                if isinstance(self.json_generator, IOBase):
                    doc = json.loads(doc.rstrip("\n").replace("'","\""))
                # Store input
                X[match_counter] = self.augment(np.array(self.x_format(doc[self.x_field]), dtype = self.x_dtype)).reshape((*self.input_shape))
                # Store target
                y[match_counter] = np.array(self.y_format(doc[self.y_field]), dtype = self.y_dtype)
                match_counter += 1
                if match_counter == self.batch_size:
                    break
            doc_counter += 1

        return X, self.__binarize(y)

    def __binarize(self, y):
        'Returns labels in binary NumPy array'
        if self.n_classes != None:
            return np.array([[1 if y[i] == j else 0 for j in range(self.n_classes)]
                            for i in range(y.shape[0])])
        else:
            return y