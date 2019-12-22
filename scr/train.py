###Import Libraries
import librosa
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import argparse
import numpy as np
import pickle
import itertools
#keras
import matplotlib.pyplot as plt
import keras
from keras.utils import to_categorical, plot_model
from keras.models import Model
from keras.callbacks import ModelCheckpoint, TensorBoard
import time
#model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from utils import *
from models import *
import config as cf

#from imblearn.over_sampling import SMOTE

parser=argparse.ArgumentParser()
parser.add_argument('--train_folder',type=str,default='./train_npy',help='folder conclude file .npy features')
parser.add_argument('--valid_folder',type=str,default='./valid_npy',help='folder conclude file .npy features')
parser.add_argument('--test_folder',type=str, default='./test_npy',help='folder conclude file .npy features')
parser.add_argument('--save_dir',type=str,default='saved_models',help='name folder to save weight and model file')
parser.add_argument('--num_class',type=int,default=4,help='num classed to classify')
parser.add_argument('--epochs',type=int,default=20,help='num epochs to training')	

if __name__=='__main__':
    flags=parser.parse_args()

    save_path = os.path.join(os.getcwd(), flags.save_dir)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    logs_path=os.path.join(save_path,'logs')
    if not os.path.isdir(logs_path):
        os.makedirs(logs_path)
    if not os.path.isdir(os.path.join(save_path,'picture')):
        os.makedirs(os.path.join(save_path,'picture'))
    if not os.path.isdir(os.path.join(save_path,'net')):
        os.makedirs(os.path.join(save_path,'net'))

    print("Getting model................................")	
    #get train,test,val
    #over_sampling=flags.over_sampling
    #if (over_sampling==True):
    #        org_x_train=x_train.shape
    #        x_train=np.reshape(x_train,(x_train[1].shape,x_train[2].shape))
    #        smote=SMOTE()
    #        x_train,y_train=smote.fit_sample(x_train,y_train)
    #        x_train=np.reshape(x_train, org_x_train)
    #x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=flags.train_size)
    split = True
    if (split == False ):
            #x_val , y_val = get_data(data_folder= flags.valid_folder, split = split)
            x_test , y_test = get_data(data_folder= flags.test_folder, split = split)
            x_train, y_train = get_data(data_folder= flags.train_folder, split = split)
    else:
            x_train, x_val, y_train, y_val = get_data(data_folder= flags.train_folder, split = split)
            x_test, y_test = get_data(flags.test_folder, False)
    x_train = np.expand_dims(x_train, axis=-1)
    print(x_train.shape)
    x_val = np.expand_dims(x_val, axis= -1)
    x_test = np.expand_dims(x_test, axis =-1)
    le=LabelEncoder()
    y_train = to_categorical(le.fit_transform(y_train))
    y_val = to_categorical(le.fit_transform(y_val))
    y_test = to_categorical(le.fit_transform(y_test))
    #return x_train, y_train, x_val, y_val, x_test, y_test
    print("Loading data from saved folder.............")
    num_class=flags.num_class
    model=get_model(num_class=num_class, x_train=x_train)
    model.summary()

    print("Save architecture model.......................")
    plot_model(model,show_shapes = True, to_file = os.path.join(save_path,'picture','model_architecture.jpg'))
    model_path = os.path.join(save_path,cf.best_model)
    save_best = ModelCheckpoint(model_path, save_best_only = True, monitor='val_loss', mode= 'min')
    name = "ser_{}".format(time.time())
    tensorboard = TensorBoard(log_dir = os.path.join(logs_path, name))
    callback = [save_best,tensorboard]

    print("Training model................................")
    history=model.fit(x_train, y_train, batch_size=cf.batch_size, epochs=flags.epochs, validation_data=(x_val, y_val),callbacks = callback)
    plot_history(history, os.path.join(save_path,'picture'))
    print("Saving model..................................")

    save_model(model, os.path.join(save_path,'net'), cf.model_name, cf.weight_name)
    ###val model
    print("Validating model..............................")
    print("For validation set:")
    full_multiclass_report(model,
                           x_val,
                           y_val,
                           le.inverse_transform(np.arange(num_class)),
                           saved_dirs = os.path.join(save_path,'picture'))
    print("For test set:")
    #full_multiclass_report(model,
    #                      x_test,
    #                      y_test,
    #                     le.inverse_transform(np.arange(num_class)),saved_dirs=save_path)


