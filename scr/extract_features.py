import librosa
import os
import argparse
import numpy as np
import pickle
from tqdm import tqdm
import tensorflow as tf
from keras.utils import to_categorical
from utils import *
import config as cf

def save_data_to_array(scr_folder, dst_folder, sr, duration, win_length,stride_length,n_mfcc,global_feat):
    labels = os.listdir(scr_folder)
    labels.sort()
    for label in labels:
        mfcc_vectors = []
        wavfiles = [scr_folder +'/'+ label + '/' + wavfile for wavfile in os.listdir(scr_folder + '/' + label)]
        for wavfile in tqdm(wavfiles, "Saving vectors of label - '{}'".format(label)):
            feature = extract_feat(wavfile, sr, duration, win_length,stride_length,n_mfcc,global_feat)
            mfcc_vectors.append(feature)
        if not os.path.isdir(dst_folder):
            os.makedirs(dst_folder)
        np.save(os.path.join(dst_folder,label + '.npy'), mfcc_vectors)
    print('labels:', labels)
if __name__ == '__main__':
  parser=argparse.ArgumentParser()
  parser.add_argument('--scr_folder',type=str,default='./train',help='Data folder to extract')
  parser.add_argument('--dst_folder',type=str,default='./test_npy',help='folder to save file extracted')
  parser.add_argument('--sr',type=int,default=16000,help='samplerate') 
  parser.add_argument('--duration',type=float,default=2,help='duration of the wave file to train')
  flags=parser.parse_args()
  save_data_to_array(flags.scr_folder,flags.dst_folder, flags.sr, flags.duration, cf.win_length, cf.stride_length, cf.n_mfcc, cf.global_feat)

