#!/usr/local/anaconda3/bin/python

from time import time
import json
import yaml

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
    'Clean data'
    #faceinfo = mat2py(doc[input_fields[0]])
    #faceinfo.extend([x**2 for x in faceinfo])
    #return py2mat(faceinfo)+"\t"+str(doc[output_fields[0]])
    return doc

'''
# config_db.yml
username: "<username>"
password: "<password>"
host: "<hostname>"
port: "<port>"
dbname: "<name of db>"
auth: "?authMechanism=SCRAM-SHA-1"
dbcoll: "<name of collection>"
query: 
  <query_field_1>: <query_value_1>
  <query_field_2>: <query_value_2>
            .
            .
            .
projection:
  <proj_field_1>: <proj_value_1>
  <proj_field_2>: <proj_value_2>
            .
            .
            .
hint:
  <hint_field>: <hint_value>
batch_size: <batch size>
'''

if __name__ == "__main__":
    # Load database keyword arguments
    with open("/Users/ross/Dropbox/Research/MLearn/config_db.yml",'r') as stream:
        try:
            db_kwargs = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # File info
    file_path = "/Users/ross/Dropbox/Research/MLearn"
    file_name = db_kwargs['dbcoll']

    # Open database
    client = MongoClient("mongodb://"+db_kwargs['username']+":"+db_kwargs['password']+"@"+['host']+":"+['port']+"/"+['dbname']+['auth'])
    db = client[db_kwargs['dbname']]
    coll = db[db_kwargs['dbcoll']]

    # Get database cursor
    curs = coll.find(db_kwargs['query'], db_kwargs['projection'], batch_size = db_kwargs['batch_size'], no_cursor_timeout = True, allow_partial_results = True).hint(list(db_kwargs['hint'].items()))

    # Loop, process, and write to json file
    count = 1
    with open(file_path+"/"+file_name+".json","w") as json_stream, open(file_path+"/"+file_name+".log","w") as log_stream:
        start_time = time()
        curr_time = start_time
        for doc in curs:
            # Clean data in document
            json_string = clean_data(doc, input_fields, output_fields)
            # Print cleaned data to json file
            print(json_string, end = '\n', file = json_stream, flush = True)
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