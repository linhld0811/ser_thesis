###Import Libraries
import librosa
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import argparse
import numpy as np
#keras
import keras
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau, TensorBoard
from tqdm import tqdm
#sklearn
#model
from sklearn.model_selection import train_test_split,StratifiedKFold, KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import LabelEncoder

#metric
from sklearn.metrics import confusion_matrix 
#imbalance data
from utils import *
from models import *
import time
###Load data
# folder save ndarray features (.npy extension)
kfold = 5
epochs = 50

def load_data_kfold(k,x_train,y_train):
    folds = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=42).split(x_train, y_train))
    return folds, x_train, y_train

def get_callbacks(path, name_weights):
    mcp_save = ModelCheckpoint(name_weights, save_best_only=True, monitor='val_loss', mode='min')
    #reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=patience_lr, verbose=1, min_delta=1e-4, mode='min')
    name = "ser_{}".format(time.time())
    tensorboard = TensorBoard(log_dir = os.path.join(path,'logs', name))
    return list([mcp_save, tensorboard])

parser=argparse.ArgumentParser()
parser.add_argument('--data_folder',type=str,default='./train_npy',help='folder conclude file .npy features')
parser.add_argument('--num_class',type=int,default=4,help='num classed to classify')
parser.add_argument('--save_dir',type=str,default='saved_models',help='name folder to save weight and model file')
if __name__=='__main__':
    flags=parser.parse_args()
    save_ckpt = os.path.join(os.getcwd(), flags.save_dir)
    if not os.path.isdir(save_ckpt):
        os.makedirs(save_ckpt)
    if not os.path.isdir(os.path.join(save_ckpt,'net')):
        os.makedirs(os.path.join(save_ckpt,'net'))
    split = False
    print("Loading data from saved folder.............")
    x_train , y_train = get_data(data_folder= flags.data_folder, split = split)
    x_train=np.expand_dims(x_train, axis=-1)
    print("Loading k-fold..............................")
    folds, X_train, y_train = load_data_kfold(kfold, x_train, y_train)
    print("Training k-fold.............................")
    for j, (train_idx, val_idx) in enumerate(folds):
        print('\nFold ',j)
        X_train_cv = X_train[train_idx]
        y_train_cv = y_train[train_idx]
        X_valid_cv = X_train[val_idx]
        y_valid_cv= y_train[val_idx]
        le=LabelEncoder()
        y_train_cv = to_categorical(le.fit_transform(y_train_cv))
        y_valid_cv = to_categorical(le.fit_transform(y_valid_cv))
        name_weights = os.path.join(save_ckpt,'net',"final_model_fold" + str(j) + "_weights.h5")
        callbacks = get_callbacks(path = save_ckpt,name_weights = name_weights)
        model=get_model(num_class = flags.num_class, x_train = x_train)
        history = model.fit(
                    X_train_cv,y_train_cv,
                    epochs = epochs,
                    batch_size = cf.batch_size,
                    shuffle = True,
                    verbose = 1,
                    validation_data = (X_valid_cv, y_valid_cv),
                    callbacks = callbacks
        )
        if not os.path.isdir(os.path.join(save_ckpt,'picture','fold'+str(j))):
            os.makedirs(os.path.join(save_ckpt,'picture','fold'+str(j)))
        plot_history(history,os.path.join(save_ckpt,'picture','fold'+str(j)) )
        print(model.evaluate(X_valid_cv, y_valid_cv))
