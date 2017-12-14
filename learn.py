import numpy as np
import json
from glob import iglob

from DataGenerator import DataGenerator, X_format, augment_features

from pymongo import MongoClient

from keras.models import Sequential
from keras.layers import Dense, Lambda, Flatten
from keras.callbacks import Callback
from keras import backend as K

class PrintBatch(Callback):
    'Class that defines a log function for model.fit_generator callbacks.'
    def on_batch_end(self, epoch, logs = {}):
        print(logs)

def train_validate_test(n_inputs, test_frac = 0.2, validate_frac = 0.2):
    'Split inputs into training, validation, and test sets'
    indices = list(range(n_inputs))
    np.random.shuffle(indices)

    n_test = int(test_frac * n_inputs)
    n_validate = int(validate_frac * (n_inputs - n_test))
    n_train = n_inputs - n_validate - n_test

    train_indices, validate_indices, test_indices = indices[:n_train], indices[n_train:(n_train + n_validate)], indices[(n_train + n_validate):]

    result = {'train': train_indices,
              'validation': validate_indices,
              'test': test_indices}

    return(result)

def count_inputs(file_pattern, x_field):
    'Count inputs'
    n_inputs = 0
    input_shape = []
    for file_name in iglob(file_pattern):
        with open(file_name, 'r') as file_stream:
            for line in file_stream:
                doc = json.loads(line.rstrip("\n").replace("'","\""))
                X = augment_features(X_format(doc[x_field]))
                for i in range(len(X.shape)):
                    if i == len(input_shape):
                        input_shape += [X.shape[i]]
                    elif X.shape[i] > input_shape[i]:
                        input_shape[i] = X.shape[i]
                n_inputs += 1
    
    return n_inputs, tuple(input_shape)

def build_model(input_shape):
    'Design model'
    model = Sequential()

    model.add(Flatten(input_shape = input_shape))
    model.add(Dense(units = 1000, activation = 'sigmoid'))
    model.add(Dense(units = 10, activation = 'tanh'))
    model.add(Dense(units = 100, activation = 'sigmoid'))
    model.add(Dense(units = 1, activation = 'sigmoid'))

    model.compile(loss = 'binary_crossentropy', optimizer = 'sgd', metrics = ['accuracy'])

    return model

if __name__ == "__main__":
    # DataGenerator positional arguments
    file_pattern = "/Users/ross/Dropbox/Research/MLearn/*.json"
    fields = ("NFORM2SKEL", "FACETNREGTRIANG")

    # DataGenerator keyword arguments
    data_gen_kwargs = {'batch_size': 32,
                       'n_classes': None,
                       'shuffle': True,
                       'seed': 0}

    # Count inputs and get max dimensions
    n_inputs, input_shape = count_inputs(file_pattern, fields[0])

    # Split inputs
    id_partition = train_validate_test(n_inputs)

    # Initialize generators
    training_generator = DataGenerator(file_pattern, input_shape, fields, **data_gen_kwargs).generate(id_partition['train'])
    validation_generator = DataGenerator(file_pattern, input_shape, fields, **data_gen_kwargs).generate(id_partition['validation'])

    # Build model
    model = build_model(input_shape)
    pb = PrintBatch()

    # Train model on dataset
    model.fit_generator(generator = training_generator,
                        steps_per_epoch = len(id_partition['train']) // data_gen_kwargs['batch_size'],
                        validation_data = validation_generator,
                        validation_steps = len(id_partition['validation']) // data_gen_kwargs['batch_size'],
                        epochs = 20,
                        verbose = 2,
                        callbacks = [pb],
                        class_weight = None,
                        workers = 1,
                        shuffle = True)