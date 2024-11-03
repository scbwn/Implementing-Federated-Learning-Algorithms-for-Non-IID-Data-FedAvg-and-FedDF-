import numpy as np
import tensorflow as tf
import random as rn
np.random.seed(1)
rn.seed(2)
tf.random.set_seed(3)
import os

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, accuracy_score
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import CategoricalCrossentropy, KLDivergence
from tensorflow.keras.layers import Layer, Input, Dense, Dropout, BatchNormalization, Activation, Add, Multiply, Lambda
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, UpSampling2D, GlobalAveragePooling2D
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Normalization, Resizing, RandomCrop, RandomFlip
from tensorflow.keras.activations import softmax
from tensorflow.keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from copy import deepcopy
import time
import gc

N=20

l_round=20 # Number of local rounds each client runs
g_round=20 # Number of distillation rounds at the server for FedDF
comm_round=50 # Number of total communication rounds

alpha=0.1 # dirichilet parameter controlling data heterogeneity
print(alpha)

def set_seed_TF2(seed):
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    rn.seed(seed)
    
def split_data(x_train, y_train, N, num_of_classes, s):
    client_y_list=[]
    for i in range(num_of_classes):
        x_cls=x_train[y_train[:,i]==1]
        y_cls=y_train[y_train[:,i]==1]
        chunk_list = np.floor(s[i]*np.sum(y_train[:,i])).astype(int)
        x_list=tf.split(x_cls[:np.sum(chunk_list)], chunk_list, axis=0)
        y_list=tf.split(y_cls[:np.sum(chunk_list)], chunk_list, axis=0)
        if i==0:
            client_x_list=[]
            client_y_list=[]
            for j in range(N):
                client_x_list.append(x_list[j])
                client_y_list.append(y_list[j])
        else:
            for j in range(N):
                client_x_list[j]=np.vstack([client_x_list[j], x_list[j]])
                client_y_list[j]=np.vstack([client_y_list[j], y_list[j]])

    return client_x_list, client_y_list
    
def resnet_layer(x,
                 num_filters,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True):

    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal')
    x = conv(x)
    if batch_normalization:
        x = BatchNormalization()(x)
    if activation is not None:
        x = Activation(activation)(x)
    return x

def building_block(x, filters=16, depth=8):
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    num_filters = filters
    num_res_blocks = int((depth - 2) / 6)

    x = resnet_layer(x, num_filters=num_filters)
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:
                strides = 2  # downsample
            y = resnet_layer(x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                pad_dim=num_filters-x.shape[-1]
                paddings=tf.constant([[0, 0,], [0, 0], [0, 0], [pad_dim-pad_dim//2, pad_dim//2]])
                x = tf.pad(x[:, ::2, ::2, :], paddings, mode="CONSTANT")
            x = Add()([x, y])
            x = Activation('relu')(x)
        num_filters*=2
    x = GlobalAveragePooling2D()(x)
    return x

def create_model(d, num_of_classes):
    set_seed_TF2(100)
    inp=Input(shape=(None,None,d[2]))
    flat_inp = building_block(inp)
    op=Dense(num_of_classes, activation='softmax')(flat_inp)
    nn = Model(inputs=inp, outputs=op)
    return nn

# Choice of Federated Learning Algorithm: fedavg / feddf
alg = 'fedavg'
print(alg)

with tf.device("CPU"):
    # Dataset Preprocessing
    d=(32,32,3)
    num_of_classes=10
    from tensorflow.keras.datasets import cifar10, cifar100
    
    # Private dataset of clients
    (x_train, y_train), (x_test, y_test)=cifar10.load_data()
    
    # Reference dataset at the server for ensemble distillation
    (x_ref, y_ref), (_, _)=cifar100.load_data()
    ref_num_of_classes=100

    x_train=x_train.astype(np.float32)/255
    x_test=x_test.astype(np.float32)/255
    x_ref=x_ref.astype(np.float32)/255

    # Scaling
    mean=x_train.mean((0,1,2))
    std=x_train.std((0,1,2))
    x_train=(x_train-mean)/std
    x_test=(x_test-mean)/std
    x_ref=(x_ref-mean)/std

    y_train=y_train.reshape(-1,1)
    y_test=y_test.reshape(-1,1)
    y_ref=y_ref.reshape(-1,1)

    # One-hot Encoding
    from tensorflow.keras.utils import to_categorical
    y_train=to_categorical(y_train)
    
    y_train=y_train.astype(np.float32)
    y_ref=y_ref.astype(np.float32)

    ref_size=5000
    x_ref=x_ref[:ref_size]
    y_ref=y_ref[:ref_size]

    #validation split
    x_val = x_train[:1000]
    y_val = y_train[:1000]
    x_train = x_train[1000:]
    y_train = y_train[1000:]

    alpha_list = [alpha] * N
    np.random.seed(1)
    s = np.random.dirichlet(alpha_list, num_of_classes)
    x_train_list, y_train_list = split_data(x_train, y_train, N, num_of_classes, s)

# Client weights
D_mat=np.zeros([N, num_of_classes])
for n in range(N):
    D_mat[n]=np.sum(y_train_list[n], axis=0)

# Create model
model_choice=create_model(d, num_of_classes)
            
# Federated Learning Algorithms
if alg=='fedavg':
    from fedlib import FedAvg
    alg_choice=FedAvg(N, l_round, comm_round, model_choice)
    global_model=alg_choice.train_model(D_mat, x_train_list, y_train_list, x_test, y_test)
if alg=='feddf':
    from fedlib import FedDF
    alg_choice=FedDF(N, l_round, g_round, comm_round, model_choice)
    global_model=alg_choice.train_model(D_mat, x_train_list, y_train_list, x_ref, x_test, y_test, x_val, y_val)
