###Import Libraries
import librosa
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import argparse
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.utils import to_categorical
from tqdm import tqdm
#sklearn
#model
from sklearn.preprocessing import LabelEncoder
#metric
from sklearn.metrics import confusion_matrix
from utils import *
import config as cf
if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--path_wav',type=str,default='./test_npy',help='path to wav file')
    parser.add_argument('--saved_dirs', type=str,default='./saved_models',help='folder save file trained')
    parser.add_argument('--duration', type=float,default=2)
    parser.add_argument('--sr', type=float,default=16000,help='samplerate')
    flags=parser.parse_args()
    labels=['angry', 'neutral']

    print("Loaded model from save_path")
    loaded_model=load_model(flags.saved_dirs,cf.model_name, cf.weight_name)
    # evaluate loaded model on test data
    opt = keras.optimizers.Adadelta()
    loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    print("Done!")

    print("Extract feature....................................")
    x_test=extract_feat(flags.path_wav, flags.sr, flags.duration, cf.win_length, cf.stride_length, cf.n_mfcc, cf.global_feat)
    print("Done!")
    x_test = np.expand_dims(x_test, axis=-1)
    x_test = np.expand_dims(x_test, axis =0)
    preds=loaded_model.predict(x_test, verbose=0)
    pred_num=np.argmax(preds)
    prediction=labels[pred_num]
    print("Emotion of wavfile is: %s" % prediction)
