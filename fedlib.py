import numpy as np
import tensorflow as tf
import random as rn
np.random.seed(1)
rn.seed(2)
tf.random.set_seed(3)
import os

from sklearn.metrics import accuracy_score
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential, clone_model
from tensorflow.keras.losses import CategoricalCrossentropy, KLDivergence, MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from copy import deepcopy
import time
import gc

batch_size=128
loss_fn = CategoricalCrossentropy()
optimizer=Adam(decay=1E-4)

def custom_T(x,T=1):
    return tf.math.softmax(tf.math.log(x+1E-15)/T)

def scale_model_weights(weight, scalar):
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final

def sum_scaled_weights(scaled_weight_list):
    avg_grad = list()
    #get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)
    return avg_grad


class FedAvg:
    def __init__(self, N, l_round, comm_round, model_choice):
        self.N=N
        self.l_round=l_round
        self.comm_round=comm_round
        self.model_choice = model_choice
    def train_model(self, D_mat, x_train_list, y_train_list, x_test, y_test):
        # Instantiate global_model and scaling weights for all clients
        scaling_factor_list=D_mat.sum(1)/D_mat.sum()
        acc_list=[]
        global_model=clone_model(self.model_choice)

        for t in range(self.comm_round):
            print("\nGlobal round %d" % (t,))
            optimizer.learning_rate=1e-3
            scaled_weight_list=[]
            for n in range(self.N):
                print("\nClient %d" % (n,))
                local_model=clone_model(self.model_choice)
                local_model.set_weights(global_model.get_weights())
                # Data generator for training data
                from tensorflow.keras.preprocessing.image import ImageDataGenerator
                train_generator = ImageDataGenerator()

                # Generate training batches
                train_batches = train_generator.flow(x_train_list[n], y_train_list[n], batch_size=batch_size)

                for epoch in range(self.l_round):
                    gc.collect()
                    print("\nStart of epoch %d" % (epoch,))

                    # Iterate over the batches of the dataset.
                    for step in range(len(x_train_list[n])//batch_size):
                        (x_batch, y_batch) = next(train_batches)

                        # Open a GradientTape for auto-differentiation.
                        tf.random.set_seed(3)
                        with tf.GradientTape() as tape:

                            # Run the forward pass 
                            y_pred = local_model(x_batch, training=True)  # predictions for this minibatch

                            # Compute the loss value for this minibatch.
                            loss_value = loss_fn(y_batch, y_pred)

                        # Retrieve the gradients
                        grads = tape.gradient(loss_value, local_model.trainable_weights)

                        # Run one step of minibatch stochastic gradient descent 
                        optimizer.apply_gradients(zip(grads, local_model.trainable_weights))

                        # Log every 10 batches.
                        if step % 10 == 0:
                            print(
                                "Training loss (for one batch) at step %d: %.4f"
                                % (step, float(loss_value))
                            )
                            print("Seen so far: %s samples" % ((step + 1) * batch_size))

                scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor_list[n])
                scaled_weight_list.append(scaled_weights)


            print("\nServer")
            # average weights
            average_weights = sum_scaled_weights(scaled_weight_list)

            # update global model
            global_model.set_weights(average_weights)

            y_pred_test=global_model.predict(x_test)
            print('Test Classification')
            print(accuracy_score(y_test, np.argmax(y_pred_test,1)))
            acc_list.append(accuracy_score(y_test, np.argmax(y_pred_test,1)))
            print(acc_list)
        return global_model

    
class FedDF:
    def __init__(self, N, l_round, g_round, comm_round, model_choice):
        self.N=N
        self.l_round=l_round
        self.g_round=g_round
        self.comm_round=comm_round
        self.model_choice = model_choice
    def train_model(self, D_mat, x_train_list, y_train_list, x_ref, x_test, y_test, x_val, y_val):
        # Instantiate global_model and scaling weights for all clients
        scaling_factor_list=D_mat.sum(1)/D_mat.sum()
        pre_dist_acc_list=[]
        acc_list=[]
        global_model=clone_model(self.model_choice)
        
        for t in range(self.comm_round):
            print("\nGlobal round %d" % (t,))
            optimizer.learning_rate=1e-3
            scaled_weight_list=[]
            scaled_pred_list=[]
            for n in range(self.N):
                print("\nClient %d" % (n,))
                local_model=clone_model(self.model_choice)
                local_model.set_weights(global_model.get_weights())
                # Data generator for training data
                from tensorflow.keras.preprocessing.image import ImageDataGenerator
                train_generator = ImageDataGenerator()

                # Generate training batches
                train_batches = train_generator.flow(x_train_list[n], y_train_list[n], batch_size=batch_size)

                for epoch in range(self.l_round):
                    gc.collect()
                    print("\nStart of epoch %d" % (epoch,))

                    # Iterate over the batches of the dataset.
                    for step in range(len(x_train_list[n])//batch_size):
                        (x_batch, y_batch) = next(train_batches)

                        # Open a GradientTape for auto-differentiation.
                        tf.random.set_seed(3)
                        with tf.GradientTape() as tape:

                            # Run the forward pass 
                            y_pred = local_model(x_batch, training=True)  # predictions for this minibatch

                            # Compute the loss value for this minibatch.
                            loss_value = loss_fn(y_batch, y_pred)

                        # Retrieve the gradients
                        grads = tape.gradient(loss_value, local_model.trainable_weights)

                        # Run one step of minibatch stochastic gradient descent
                        optimizer.apply_gradients(zip(grads, local_model.trainable_weights))

                        # Log every 10 batches.
                        if step % 10 == 0:
                            print(
                                "Training loss (for one batch) at step %d: %.4f"
                                % (step, float(loss_value))
                            )
                            print("Seen so far: %s samples" % ((step + 1) * batch_size))

                scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor_list[n])
                scaled_weight_list.append(scaled_weights)
                scaled_preds = custom_T(local_model.predict(x_ref))*scaling_factor_list[n]
                scaled_pred_list.append(scaled_preds)

            print("\nServer")
            # average weights
            average_weights = sum_scaled_weights(scaled_weight_list)
            average_preds = np.sum(scaled_pred_list, axis=0)
            
            # update global model
            global_model.set_weights(average_weights)

            y_pred_test=global_model.predict(x_test)
            print('Test Classification')
            print(accuracy_score(y_test, np.argmax(y_pred_test,1)))
            pre_dist_acc_list.append(accuracy_score(y_test, np.argmax(y_pred_test,1)))
            # Generate reference batches
            dist_batches = train_generator.flow(x_ref, average_preds, batch_size=batch_size)
            
            callback = EarlyStopping(monitor='val_loss', patience=5)
            global_model.compile(optimizer=optimizer, loss=loss_fn)
            tf.random.set_seed(3)
            global_model.fit(dist_batches, batch_size=batch_size, epochs=self.g_round, verbose=2,
                                                callbacks=[callback], validation_data=(x_val, y_val))

            y_pred_test=global_model.predict(x_test)
            print('Test Classification')
            print(accuracy_score(y_test, np.argmax(y_pred_test,1)))
            acc_list.append(accuracy_score(y_test, np.argmax(y_pred_test,1)))
            print(pre_dist_acc_list)
            print(acc_list)
            
        return global_model
    
