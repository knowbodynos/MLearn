import numpy as np

from DataGenerator import DataGenerator

from pymongo import MongoClient

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import Callback

class PrintBatch(Callback):
    def on_batch_end(self, epoch, logs={}):
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

def get_json_generator(json_gen_kwargs):
    if 'file_name' in json_gen_kwargs.keys():
        json_generator = open(json_gen_kwargs['file_name'], 'r')
        n_inputs = 0
        for line in json_generator:
            n_inputs += 1
        json_generator.seek(0,0)
    else:
        client = MongoClient("mongodb://"+json_gen_kwargs['username']+":"+json_gen_kwargs['password']+"@"
                             +json_gen_kwargs['host']+":"+json_gen_kwargs['port']+"/"+json_gen_kwargs['dbname']
                             +json_gen_kwargs['auth'])
        db = client[json_gen_kwargs['dbname']]
        coll = db[json_gen_kwargs['dbcoll']]
        json_generator = coll.find(json_gen_kwargs['query'], json_gen_kwargs['projection'],
                                  batch_size = json_gen_kwargs['batch_size'],
                                  no_cursor_timeout = True,
                                  allow_partial_results = True).hint(json_gen_kwargs['hint'])
        n_inputs = json_generator.count()

    return n_inputs, json_generator

def count_inputs(file_name):
    with open(file_name,'r') as file_stream:
        n_inputs = 0
        for line in file_stream:
            n_inputs += 1

    return n_inputs

def augment_data(x_val):
    return np.append(x_val, x_val ** 2)

def x_format(x_val):
    return eval(str(x_val).replace(" ","").replace("{","[").replace("}","]"))

def build_model(input_shape):
    'Design model'
    model = Sequential()

    model.add(Dense(units = 1000, activation = 'sigmoid', input_shape = input_shape))
    model.add(Dense(units = 100, activation = 'tanh'))
    model.add(Dense(units = 1, activation = 'sigmoid'))

    model.compile(loss = 'binary_crossentropy', optimizer = 'sgd', metrics = ['accuracy'])

    return(model)

if __name__ == "__main__":
    # Batch Size
    batch_size = 32
    epochs = 20

    # JSON Generator keyword arguments
    json_gen_kwargs = {'file_name': "/Users/ross/Dropbox/Research/MLearn/FACET.json"}
    '''
    json_gen_kwargs = {'username': "manager",
                       'password': "toric",
                       'host': "129.10.135.170",
                       'port': "27017",
                       'dbname': "MLEARN",
                       'auth': "?authMechanism=SCRAM-SHA-1",
                       'dbcoll': "FACET",
                       'query': {'facetfinetriangsMARK':True},
                       'projection': {'_id':0,'FACEINFO':1,'FACETNREGTRIANG':1},
                       'hint': list({'facetfinetriangsMARK':1}.items()),
                       'batch_size': epochs * batch_size}
    '''

    # Obtain generator and count inputs
    n_inputs, json_generator = get_json_generator(json_gen_kwargs)

    # DataGenerator arguments
    #file_name = "/Users/ross/Dropbox/Research/MLearn/FACET.json"
    input_shape = (2 * 17,)
    x_field = "FACEINFO"
    y_field = "FACETNREGTRIANG"

    # DataGenerator keyword arguments
    data_gen_kwargs = {'batch_size': batch_size,
              'n_classes': None,
              'x_dtype': int,
              'y_dtype': int,
              'x_format': x_format,
              'y_format': lambda y: y,
              'augment': augment_data,
              'shuffle': True}

    # Count inputs
    #n_inputs = count_inputs(file_name)

    # Split inputs
    id_partition = train_validate_test(n_inputs)

    # Initialize generators
    training_generator = DataGenerator(json_generator, input_shape, x_field, y_field, **data_gen_kwargs).generate(id_partition['train'])
    validation_generator = DataGenerator(json_generator, input_shape, x_field, y_field, **data_gen_kwargs).generate(id_partition['validation'])

    # Build model
    model = build_model(input_shape)
    pb = PrintBatch()

    # Train model on dataset
    model.fit_generator(generator = training_generator,
                        steps_per_epoch = len(id_partition['train'])//data_gen_kwargs['batch_size'],
                        validation_data = validation_generator,
                        validation_steps = len(id_partition['validation'])//data_gen_kwargs['batch_size'],
                        epochs = epochs,
                        verbose = 2,
                        callbacks = [pb],
                        class_weight = None,
                        workers = 1,
                        shuffle = True)

    json_generator.close()