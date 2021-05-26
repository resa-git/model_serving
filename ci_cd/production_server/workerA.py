from celery import Celery

from numpy import loadtxt
import numpy as np
import pickle
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Sequential
from sklearn.metrics import r2_score
import tensorflow as tf

model_json_file = './model.json'
model_weights_file = './model.h5'
data_file = './stargazer.csv'

def load_data():
    dataset =  loadtxt(data_file, delimiter=',', skiprows=1)
    X = dataset[:,0:7]
    y = dataset[:,7]
    #y = list(map(int, y))
    #y = np.asarray(y, dtype=np.uint8)
    return X, y

def load_model():
    # load json and create model
    #json_file = open(model_json_file, 'r')
    #loaded_model_json = json_file.read()
    #json_file.close()
    #loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    #loaded_model.load_weights(model_weights_file)
    #print("Loaded model from disk")
    filename = "final_model.sav"
    loaded_model = pickle.load(open(filename, "rb"))
    return loaded_model

# Celery configuration
CELERY_BROKER_URL = 'amqp://rabbitmq:rabbitmq@rabbit:5672/'
CELERY_RESULT_BACKEND = 'rpc://'
# Initialize Celery
celery = Celery('workerA', broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)

@celery.task()
def add_nums(a, b):
   return a + b

@celery.task
def get_predictions():
    results ={}
    X, y = load_data()
    loaded_model = load_model()
    predictions = loaded_model.predict(X)
    results['y'] = y.tolist()
    results['predicted'] =[]
    #print ('results[y]:', results['y'])
    for i in range(len(results['y'])):
        #print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))
        results['predicted'].append(predictions[i])
    #print ('results:', results)
    return results

@celery.task
def get_accuracy():
    X, y = load_data()
    loaded_model = load_model()
    #loaded_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    score = r2_score(X, y)
    #print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
    return score[1]

print(get_predictions())
