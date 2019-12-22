###Import Libraries
import librosa
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#keras
import tensorflow as tf
import keras
from keras.utils import to_categorical
from tqdm import tqdm
#sklearn
#model
from sklearn.preprocessing import LabelEncoder
#metric
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import confusion_matrix 
import config as cf
from utils import *
parser=argparse.ArgumentParser()
parser.add_argument('--test_folder',type=str, default='./test_npy',help='folder conclude file .npy features')
parser.add_argument('--save_dir',type=str,default='saved_models',help='name folder to save weight and model file')
parser.add_argument('--result',type=str,default='result',help='name folder to save file results')
flags=parser.parse_args() 

x_test, y_test = get_data(flags.test_folder)
x_test = np.expand_dims(x_test, axis =-1)
le=LabelEncoder()
y_test = to_categorical(le.fit_transform(y_test))

###Predict
#load model
print("Loaded model from disk")
loaded_model=load_model(flags.save_dir,cf.model_name, cf.weight_name)
# evaluate loaded model on test data
opt = keras.optimizers.Adadelta()
loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

print("Evaluating model..............................")
score = loaded_model.evaluate(x_test, y_test, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
print("Saving prediction to csv,.....................")
preds = loaded_model.predict(x_test, 
                         batch_size=cf.batch_size, 
                         verbose=0)
print(preds)
preds1=preds.argmax(axis=1)
abc = preds1.astype(int).flatten()
predictions = (le.inverse_transform((abc)))
preddf = pd.DataFrame({'predictedvalues': predictions})
actual=y_test.argmax(axis=1)
abc123 = actual.astype(int).flatten()
actualvalues = (le.inverse_transform((abc123)))
actualdf = pd.DataFrame({'actualvalues': actualvalues})
accs=preds.max(axis=1)
accuracy = pd.DataFrame({'accuracy': accs})
dfr = actualdf.join(preddf)
finaldf = dfr.join(accuracy)
if not os.path.isdir(flags.result):
    os.makedirs(flags.result)
finaldf.to_csv(os.path.join(flags.result,'Predictions_all.csv'), index=False)

