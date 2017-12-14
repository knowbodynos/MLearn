import numpy as np
from glob import iglob
import json
#import tempfile
import sys

# Helper functions
def X_format(X):
    'Format an input value'
    return np.array(eval(str(X).replace(" ","").replace("{","[").replace("}","]")), dtype = int)

def X_unformat(X):
    'Unformat an input value'
    return str(X.tolist()).replace(" ","").replace("[","{").replace("]","}")

def augment_data(features, seed = 0, n_transforms = 5):
    'Augment the data by adding transformations on each input'
    np.random.seed(seed)

    augmented_features = []
    for n in range(n_transforms):
        # Permute the columns i.e. the lattice points
        features_augment = features[np.random.permutation(features.shape[0]),:]
        augmented_features.append(features_augment)

    return np.array(augmented_features)

if __name__ == "__main__":
    # DataGenerator positional arguments
    file_pattern = "/Users/ross/Dropbox/Research/MLearn/*.json"
    X_field = "NFORM2SKEL"

    for file_name in iglob(file_pattern):
        with open(file_name, 'r') as file_stream, open(file_name.rstrip(".json") + "_augmented.json", 'w') as new_file_stream:
            for line in file_stream:
                doc = json.loads(line.rstrip("\n").replace("'","\""))
                new_doc = doc.copy()
                doc.update({'ORIG': True})
                new_doc.update({'ORIG': False})
                #print(json.dumps(doc, separators = (',', ':')), end = '\n', file = sys.stdout, flush = True)
                print(json.dumps(doc, separators = (',', ':')), end = '\n', file = new_file_stream, flush = True)
                features = X_format(doc[X_field])
                augmented_features = augment_data(features)
                for features in augmented_features:
                    new_doc.update({X_field: X_unformat(features)})
                    #print(json.dumps(new_doc, separators = (',', ':')), end = '\n', file = sys.stdout, flush = True)
                    print(json.dumps(new_doc, separators = (',', ':')), end = '\n', file = new_file_stream, flush = True)