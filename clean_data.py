#!/usr/local/anaconda3/bin/python

from time import time
import json

#from gevent import monkey
#monkey.patch_all()
from pymongo import MongoClient

# Helper functions
def py2mat(lst):
    "Converts a Python list to a string depicting a list in Mathematica format."    
    return str(lst).replace(" ","").replace("[","{").replace("]","}")

def mat2py(lst):
    "Converts a string depicting a list in Mathematica format to a Python list."
    return eval(str(lst).replace(" ","").replace("{","[").replace("}","]"))

# Clean data function
def clean_data(doc, input_fields, output_fields):
    #faceinfo = mat2py(doc[input_fields[0]])
    #faceinfo.extend([x**2 for x in faceinfo])
    #return py2mat(faceinfo)+"\t"+str(doc[output_fields[0]])
    return doc

# Database info
username = "manager"
password = "toric"
host = "129.10.135.170"
port = "27017"
dbname = "MLEARN"
auth = "?authMechanism=SCRAM-SHA-1"
dbcoll = "FACET"
input_fields = ["FACEINFO"]
output_fields = ["FACETNREGTRIANG"]

# File info
file_path = "/Users/ross/Dropbox/Research/MLearn"
file_name = dbcoll

# Open database
client = MongoClient("mongodb://"+username+":"+password+"@"+host+":"+port+"/"+dbname+auth)
db = client[dbname]
coll = db[dbcoll]

# Query info
query = {'facetfinetriangsMARK': True}
projection = {'_id': 0}
projection.update({x: 1 for x in input_fields+output_fields})
hint = [('facetfinetriangsMARK', 1)]
batch_size = 500

# Get database cursor
curs = coll.find(query, projection, batch_size = batch_size, no_cursor_timeout = True, allow_partial_results = True).hint(hint)

# Loop, process, and write to csv file
count = 1
with open(file_path+"/"+file_name+".json","w") as csv_stream, open(file_path+"/"+file_name+".log","w") as log_stream:
    start_time = time()
    curr_time = start_time
    for doc in curs:
        # Clean data in document
        csv_string = clean_data(doc, input_fields, output_fields)
        # Print cleaned data to csv file
        print(csv_string, end = '\n', file = csv_stream, flush = True)
        # Update prev_time if end of batch
        if count % batch_size == 0:
            prev_time = curr_time
        # Update curr_time if beginning of new batch
        elif (count - 1) % batch_size == 0 and count != 1:
            curr_time = time()
            count_string = "Finished writing {count} documents.".format(count = str(count))
            time_string = "Time: t = {t} seconds, \u0394t = {dt} seconds.".format(t = str(curr_time - start_time), dt = str(curr_time - prev_time))
            # Print to screen
            print(count_string, end = '\n')
            print(time_string, end = '\n\n')
            # Print to log file
            print(count_string, end = '\n', file = log_stream, flush = True)
            print(time_string, end = '\n\n', file = log_stream, flush = True)
        count += 1