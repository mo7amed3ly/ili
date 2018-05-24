
# Use scikit-learn to grid search the learning rate and momentum
import numpy
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD

from numpy.random import seed
from tensorflow import set_random_seed
seed(1234)
set_random_seed(1234)
from pathlib import Path
import time
from os.path import isdir, join
import pickle
from scipy.io import wavfile
import numpy as np
import pandas as pd
from scipy import signal
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder

import keras

from keras.layers import Embedding,SimpleRNN,Masking,Input,Conv1D,Conv2D,Conv3D, BatchNormalization, MaxPooling1D,MaxPooling2D,AveragePooling2D,MaxPooling3D, Dense, Input, Dropout, Flatten,GRU,TimeDistributed,Bidirectional,LSTM,Input,concatenate,Activation,average,SpatialDropout1D,merge,concatenate
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.models import Sequential,Model,load_model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import glob

from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

seed(1234)
set_random_seed(1234)
maxlen=256
log_dest='gdi_teng.csv'

def get_data(train_path,dev_path):
    with open(train_path) as f:
        data=f.readlines()
        X_train,labels_train=[d.split('\t')[0] for d in data if  d.split('\t')[1].replace('\n','') in ['HIN','BRA','AWA','MAG','BHO']],[d.split('\t')[1].replace('\n','') for d in data if  d.split('\t')[1].replace('\n','') in ['HIN','BRA','AWA','MAG','BHO']]

    with open(dev_path) as f:
        data_dev=f.readlines()
        X_dev,labels_dev=[d.split('\t')[0] for d in data_dev],[d.split('\t')[1].replace('\n','') for d in data_dev]

    return X_train,labels_train,X_dev,labels_dev

'''conv_units=512   lr=0.00100   c_drop=0.30   maxlen=256   embed_len=31   conv_layers=3 embed_len=8'''
# Function to create model, required for KerasClassifier
def create_model(conv_units=128, lr=1e-4,drop=0.2,conv_layers=3):
    
    ch_input=Input(shape=(maxlen,))
    ch_embed=Embedding(len(tokenizer.word_index)+1, 128,embeddings_regularizer=keras.regularizers.l2(1e-4))(ch_input)
    drp=SpatialDropout1D(0.5)(ch_embed)
    #ch_embed=Embedding(len(tokenizer.word_index)+1, len(tokenizer.word_index)+1,embeddings_initializer='identity',trainable=False)(ch_input)
    #ch_gru=GRU(128,dropout=0.2,recurrent_dropout=0.2,return_sequences=True)(ch_embed)
    to_merge=[]
    for i in range(2,8):
        ch_conv=Conv1D(256,3,activation='relu')(drp)
        bn=BatchNormalization()(ch_conv)
        ch_pool=MaxPooling1D(maxlen-i)(bn)
        ch_flat=Flatten()(ch_pool)

        ch_drp=Dropout(0.2)(ch_flat)
        to_merge.append(ch_drp)
    mrg=concatenate(to_merge)

    
    ch_ouput=Dense(5,activation='softmax')(mrg)

    
    model=Model(inputs=[ch_input],outputs=ch_ouput)
    model.compile(optimizer=Adam(lr), loss='categorical_crossentropy', metrics=['acc'])
    with open(log_dest,'a') as f:
        f.write('conv_units=%d , lr=%.5f , c_drop=%.2f , maxlen=%d , embed_len=%d , conv_layers=%d\n'%(conv_units,
                                                                                              lr,drop,maxlen,maxlen,conv_layers))
    with open('_'+log_dest,'a') as f:
        f.write('conv_units=%d , lr=%.5f , c_drop=%.2f , maxlen=%d , embed_len=%d , conv_layers=%d\n'%(conv_units,
                                                                                              lr,drop,maxlen,maxlen,conv_layers))
    print(model.summary())
    return model
# fix random seed for reproducibility




X_train,y_train,X_dev,y_dev=get_data('train.txt','dev.txt')
print('Tokenization')
tokenizer = Tokenizer(lower=False,char_level=True,filters='0123456789')
tokenizer.fit_on_texts(X_train)
X_train_seq=tokenizer.texts_to_sequences(X_train)
X_dev_seq=tokenizer.texts_to_sequences(X_dev)
print(tokenizer.word_index)
print('Padding')
X_train=pad_sequences(X_train_seq,maxlen=maxlen,padding='post')
X_dev=pad_sequences(X_dev_seq,maxlen=maxlen,padding='post')
print(X_train[0])
print('One-hot Encoding')

encoder = LabelEncoder()
encoder.fit(y_train)
y_train = encoder.transform(y_train)
y_dev = encoder.transform(y_dev)
y_train=np.array([to_categorical(y,5) for y in y_train])

y_dev=np.array([to_categorical(y,5) for y in y_dev])
X_train_seq=np.array(X_train_seq)
X_dev_seq=np.array(X_dev_seq)

X_all=np.concatenate((X_train,X_dev))
y_all=np.concatenate((y_train,y_dev))

#X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

X_all=np.concatenate((X_train,X_dev))
y_all=np.concatenate((y_train,y_dev))


seed = 7
numpy.random.seed(seed)
idx=np.random.permutation(len(X_all))
X_all=X_all[idx]
y_all=y_all[idx]
from sklearn.model_selection import KFold


#test_fold = [-1]*len(X_train)+[1]*len(X_dev)
kf = KFold(n_splits=10)



# create model

# define the grid search parameters

conv_layers=[3]
conv_units=[256]
lr=[5e-4]
drop=[0.3]
i=0
for t,v in kf.split(X_all):
    i+=1
    X_train,y_train,X_dev,y_dev=X_all[t],y_all[t],X_all[v],y_all[v]
    #X_train_em,X_dev_em=X_all_em[t],X_all_em[v]
    model = create_model(conv_layers=4,conv_units=256,lr=5e-4,drop=0.3)
    checkpoint_dir='e-k%d'%(i)
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    check_point=keras.callbacks.ModelCheckpoint('%s/model.{epoch:02d}-{val_loss:.3f}-{val_acc:.3f}.hdf5'%(checkpoint_dir),
                                                            monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

    logger=keras.callbacks.CSVLogger(log_dest, separator=',', append=True)
    early_stoppig=keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=40, verbose=0, mode='auto')
    reduce_lr=keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5,  min_lr=5e-6)
    model.fit(X_train, y_train,validation_data=(X_dev,y_dev),callbacks=[logger,early_stoppig,reduce_lr,check_point],verbose=1,epochs=200,batch_size=128)
    
