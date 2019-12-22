import librosa
import os, random
import argparse
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import itertools
from keras.models import model_from_json
from keras.utils import to_categorical, plot_model
import config as cf
def random_segment(audio_signal, N):
    length = audio_signal.shape[0]
    if N < length:
        start = random.randint(0, length - N)
        audio_signal = audio_signal[start:start + N]
    else:
        tmp = np.zeros((N,))
        start = random.randint(0, N - length)
        tmp[start: start + length] = audio_signal
        audio_signal = tmp
    return audio_signal

def load_wav(file_path, sr, duration, win_length, stride_length):
    wave, sr = librosa.load(file_path, sr=sr)
    N = int(duration*sr)
    wave = random_segment(wave, N)
    n_frame = int((duration-win_length)/stride_length)+1
    win_sample = int(win_length*sr) 
    stride_sample = int(stride_length*sr)
    wav=np.zeros(n_frame*win_sample)
    for i in range(n_frame):
        wav[i*win_sample:(i+1)*win_sample]=wave[i*stride_sample:i*stride_sample+win_sample]
    return wav, sr

def feat_mfcc(file_path, sr, duration, win_length, stride_length, n_mfcc):
    wave, sr = load_wav(file_path, sr, duration, win_length, stride_length)
    hop_length= int(sr*win_length)
    mfcc = librosa.feature.mfcc(y = wave, sr = sr, hop_length = hop_length, n_mfcc = n_mfcc)
    return mfcc

def feat_spectral_rolloff(file_path, sr, duration, win_length, stride_length):
    wave, sr = load_wav(file_path, sr, duration, win_length, stride_length)
    hop_length= int(sr*win_length)
    rolloff = librosa.feature.spectral_rolloff(y = wave, sr = sr, hop_length = hop_length)
    return rolloff

def feat_spectral_centroid(file_path, sr, duration, win_length, stride_length):
    wave, sr = load_wav(file_path, sr, duration, win_length, stride_length)
    hop_length= int(sr*win_length)
    centroid = librosa.feature.spectral_centroid(y = wave, sr = sr, hop_length = hop_length)
    return centroid

def feat_zcr(file_path, sr, duration, win_length, stride_length):
    wave, sr = load_wav(file_path, sr, duration, win_length, stride_length)
    hop_length= int(sr*win_length)
    zcr = librosa.feature.zero_crossing_rate(y = wave, hop_length=hop_length) 
    return zcr

def feat_chroma(file_path, sr, duration, win_length, stride_length):
    wave, sr = load_wav(file_path, sr, duration, win_length, stride_length)
    hop_length= int(sr*win_length)
    chroma=librosa.feature.chroma_stft(y = wave, sr = sr, hop_length=hop_length) 
    return chroma

def feat_global(file_path, sr, duration, win_length, stride_length):
    wave, sr = load_wav(file_path, sr, duration, win_length, stride_length)
    zcr = feat_zcr(file_path, sr,duration, win_length, stride_length)
    spectral_rolloff = feat_spectral_rolloff(file_path, sr,duration, win_length, stride_length)
    spectral_centroid = feat_spectral_centroid(file_path, sr,duration, win_length, stride_length)
    chroma = feat_chroma(file_path, sr,duration, win_length, stride_length)
    global_feat = np.concatenate((zcr,spectral_rolloff,spectral_centroid,chroma),axis=0)
    return global_feat

def feat_melspectrogram(file_path, sr, duration, win_length, stride_length):
    wave, sr = load_wav(file_path, sr, duration, win_length, stride_length)
    hop_length= int(sr*win_length)
    return librosa.feature.melspectrogram(y = wave, sr = sr, hop_length=hop_length)

def extract_feat(path, sr, duration, win_length,stride_length,n_mfcc,global_feat):
    mfcc = feat_mfcc(path, sr, duration, win_length, stride_length, n_mfcc)
    #delta
    mfcc_delta=librosa.feature.delta(mfcc)
    #delta2
    mfcc_delta2=librosa.feature.delta(mfcc, order=2)   
    feature=np.concatenate((mfcc,mfcc_delta,mfcc_delta2),axis=0)
    if (global_feat == True):
        feature = np.concatenate((feature, feat_global(path, sr, duration, win_length, stride_length)), axis=0)
    return feature

def extract_mel(path, sr, duration, win_length,stride_length,n_mfcc,global_feat):
    mel = feat_melspectrogram(path, sr, duration, win_length, stride_length)
    return mel

def get_data(data_folder='./save_train',split=False):
    # Get available labels
    name_file = os.listdir(data_folder)
    labels = []
    for i in name_file:
        label = os.path.splitext(i)[0]
        labels = np.append(labels,label)
    labels.sort()
    # Getting first arrays
    X = np.load(os.path.join(data_folder,labels[0] + '.npy'))
    y=[]
    y = np.append(y, np.full(X.shape[0], fill_value= (labels[0])))
    # Append all of the dataset into one single array, same goes for y
    for i, label in enumerate(labels[1:]):
        #from file .npy
        x = np.load(os.path.join(data_folder,label + '.npy'))
        #from file .pickle
        #f = open('./file_name.pkl','rb')
        #X = pickle.load(f)
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value= (label)))
    assert X.shape[0] == len(y)
    if ( split == True ):
        return train_test_split(X, y, test_size= 0.2, random_state=25, shuffle=True)
    if ( split == False ):
        return X,y

def save_model(model, save_path, model_name, weight_name):
    model_path = os.path.join(save_path, weight_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)
    model_json = model.to_json()
    with open(os.path.join(save_path,cf.model_name), "w") as json_file:
        json_file.write(model_json)

def load_model(save_path, model_name, weight_name):
	json_file = open(os.path.join(save_path,model_name), 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights(os.path.join(save_path, weight_name))
	return loaded_model

def plot_history(history,saved_dirs):
    """
    This function to plot history of train and test progress
    """
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
    if len(loss_list) == 0:
        print('Loss is missing in history')
        return
    ## As loss always exists
    epochs = range(1,len(history.history[loss_list[0]]) + 1)
    ## Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(saved_dirs,'history_loss.png'))
    plt.clf()
    ## Accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
    for l in val_acc_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(saved_dirs,'history_accuracy.png'))
    plt.clf()
    #plt.show()

def plot_confusion_matrix(cm, classes, saved_dirs,
                          normalize=False,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title='Normalized confusion matrix'
    else:
        title='Confusion matrix'
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(saved_dirs,'confusion_matrix.png'))
    plt.clf()
    #plt.show()


def full_multiclass_report(model,
                           x,
                           y_true,
                           classes,
                           batch_size=64,
                           binary=False,
                           saved_dirs='save'):
    # 1. Transform one-hot encoded y_true into their class number
    if not binary:
        y_true = np.argmax(y_true,axis=1)
    
    # 2. Predict classes and stores in y_pred
    y_pred = model.predict_classes(x, batch_size=batch_size)

    # 3. Print accuracy score
    print("Accuracy : "+ str(accuracy_score(y_true,y_pred)))

    print("")

    # 4. Print classification report
    print("Classification Report")
    print(classification_report(y_true,y_pred,digits=5))

    # 5. Plot confusion matrix
    print("Confusion matrix")
    cnf_matrix = confusion_matrix(y_true,y_pred)
    print(cnf_matrix)
    plot_confusion_matrix(cnf_matrix,saved_dirs=saved_dirs,classes=classes)
