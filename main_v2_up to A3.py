# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 15:46:44 2021

@author: stef
"""


# -*- coding: utf-8 -*-
"""
DESCRIPTION: this file contains a simple tensorflow model
             and functions for MNIST digits dataset.
"""
# importing libraries
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt

# create a function for data preprocessing and handling testing, validation and testing set
def data_generator(filename_1, filename_2, folds_num,
                   normalization = True, standardization = True, build_barcharts = True):
    """
    filename_1: filename containing training set
    filename_2: filename containing testing set
    folds_num: number of folds for k fold cross validation
    normalization: whether to perform normalization or not
    centering: whether to perform normalization or not
    standardization: whether to perform standardization or not
    build_barcharts: create barcharts for each class quantity
    """
    train_data = np.loadtxt(filename_1, delimiter = ',', skiprows = 1) # importing train set
    test_data = np.loadtxt(filename_2, delimiter = ',', skiprows = 1) # importing test set

    train_x = train_data[:, 1:]
    train_y = train_data[:, 0]
    test_x = test_data[:, 1:]
    test_y = test_data[:, 0]

    if normalization:
        # normalize test data
        test_x = test_x / 255.
        # normalize train dat
        train_x = train_x / 255.
    if standardization:
        # center test data
        mean_test = np.mean(test_x)
        std_test = np.std(test_x)
        test_x = (test_x - mean_test) / std_test
        # center train data
        mean_train = np.mean(train_x)
        std_train = np.std(train_x)
        train_x = (train_x - mean_train) / std_train

    ratio = 1. / folds_num
    packages_x=[]
    packages_y=[]
    for i in range(0, folds_num):
        packages_x.append(train_x[int(i * ratio * train_x.shape[0]):int((i + 1) * ratio * train_x.shape[0]), :])
        packages_y.append(train_y[int(i * ratio * train_y.shape[0]):int((i + 1) * ratio * train_y.shape[0])])

    return packages_x, packages_y, test_x, test_y, mean_train, std_train, mean_test, std_test
def define_model(inputs , h, activation_hidden, activation_output, optimizer, loss_name):
    """

    """
    # specify model type
    model = tf.keras.Sequential()
    # add 1st hidden layer
    model.add(layers.Dense(h[0], activation= activation_hidden, input_shape= (inputs,)))
    # add the rest hidden layers
    for n in range(1, len(h)):
        model.add(layers.Dense(h[n], activation=activation_hidden))
    # add final layer
    model.add(layers.Dense(10, activation_output))
    
    # Training and evaluation of models can be done by using Cross-Entropy
    #(CE), but also by Mean Squared Error(MSE)

    if loss_name == "MSE":
        loss = tf.keras.losses.MSE()
    elif loss_name == "CE":
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer, loss, metrics=['accuracy'])
    model.summary()
    return model

def main_function(filename_1, filename_2, folds_num, h, activation_hidden,
                  activation_output, optimizer_name, loss_name, epochs, batch_size,lr, momentum):
    # produce data
    packages_x, packages_y, test_x, test_y, mean_train, std_train, mean_test, std_test = data_generator(filename_1, filename_2, folds_num,
                                                                                                        normalization=True, standardization=True, build_barcharts=True)
    
    models = []
    for i in range(0, folds_num):
        val_x = packages_x[i]
        val_y = packages_y[i]
        k = []
        for j in range(0, folds_num):
            if j != i:
                k.append(j)
        train_x = packages_x[k[0]]
        train_y = packages_y[k[0]]
        for j in range(1, len(k)):
            train_x = np.concatenate((train_x, packages_x[k[j]]), axis=0)
            train_y = np.concatenate((train_y, packages_y[k[j]]), axis=0)
        print(train_x.shape)
        print(val_x.shape)
        # specify model type
        model = tf.keras.Sequential()
        
        
        # Inputs for 28x28 picture
        
        # add 1st hidden layer
        model.add(layers.Dense(h[0], activation=activation_hidden, input_shape=(train_x.shape[1],)))
        # add the rest hidden layers
        for n in range(1, len(h)):
            model.add(layers.Dense(h[n], activation=activation_hidden))
        # add final layer
        model.add(layers.Dense(10, activation_output))
        if loss_name == "MSE":
            loss = 'mse'
        elif loss_name == "CE":
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=0, patience=30, verbose=1,
            mode='auto', baseline=None, restore_best_weights=False)
        if optimizer_name == 'adam':
            optimizer = tf.keras.optimizers.Adam( learning_rate=lr , beta_1=0.9, beta_2=0.999,
                                                  epsilon=1e-07, amsgrad=False,name='Adam')
        elif optimizer_name == 'SGD':
            optimizer = tf.keras.optimizers.SGD( learning_rate=lr , momentum= momentum, nesterov= False, name='SGD')
                
            
        model.compile(optimizer, loss, metrics=['accuracy'])
        model.summary()
        history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, validation_data=(val_x, val_y),
                            validation_batch_size=batch_size, callbacks=[callback])

        models.append(model)
        
        #Graphs 
        #plot training and validation loss
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        ep = range(1, len(loss) + 1) # ranging epochs from one to length of loss for graphs
        fig, ax = plt.subplots(1)
        ax.plot(ep, loss, 'g.', label='Training Loss')
        ax.plot(ep, val_loss, 'b', label='Validation Loss')
        ax.set_title('Training and Validation Loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        leg = ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize=7)
        leg.get_frame().set_alpha(0.5)
        fig.show()
        
        #compare how the model performs on the test dataset
        #text_images=text_x
        #test_labels=test_y
        test_loss, test_acc = model.evaluate(test_x,  test_y, verbose=2)
        print('\nTest accuracy:', test_acc)
        
        #Make predictions-Softmax
        probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

        predictions = probability_model.predict(test_x)
        #First prediction
        print (predictions[0])
        #You can see which label has the highest confidence value
        print (np.argmax(predictions[0]))
        #test labels
        print(test_y [0])
        

if __name__ == "__main__":
    
    #names for train and test set
    train_filename = 'mnist_train.csv'
    test_filename = 'mnist_test.csv'
    h = [397,397] #Defining hidden layers and neurons for each layers. For example [128]=one layer with 128 neurons, [128,128]= two hidden layers with 128 each.
    folds_num = 5
    activation_hidden = 'relu' #Activation function for every hidden layer.
    activation_output = None   #Activation function for the output layer.
    optimizer = 'SGD'          # Define optimizer, options 'SGD' , 'adam'.
    loss = 'CE'                # Define cost function, options 'CE'(Cross entropy), 'MSE' (Mean Squared Error).
    epochs = 10
    batch_size = 8
    lr = 0.001 #Learning rate
    momentum=0.6
    

    main_function(train_filename, test_filename, folds_num, h, activation_hidden,
                  activation_output, optimizer, loss, epochs, batch_size, lr, momentum,r)

