import pandas as pd
import csv
import numpy as np
from keras import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler

from keras import backend as K
from keras import initializers
from keras import optimizers
from keras import regularizers
from sklearn.metrics import accuracy_score
import tensorflow as tf
from sklearn.model_selection import train_test_split
from numpy import linalg as LA

class nn_em:
    def __init__(self):
        print("model initialized")

    def my_init(self, shape):
        value = np.random.random(shape)
        return K.variable(value)

    def init_probabilities(self,n):
        # initialize probability z_i (item's quality) randomly
        p_z_i = np.random.randint(2, size=(n, 1)).astype(float)
        return p_z_i, 1 - p_z_i


    def define_nn(self,n_neurons, hidden,m,nb_hidden_layer,learning_rate):
        classifier = Sequential()
        if hidden == True:
            # First Hidden Layer
            layer0 = Dense(n_neurons, activation='sigmoid', kernel_initializer=initializers.random_normal(stddev=0.03, seed=98765), input_dim=m)
            classifier.add(layer0)
            nb = 1
            while (nb < nb_hidden_layer):
                layer_nb = Dense(n_neurons, activation='sigmoid', kernel_initializer=initializers.random_normal(stddev=0.03, seed=98765))
                classifier.add(layer_nb)
                nb += 1
        # Output Layer
        layer1 = Dense(1, activation='sigmoid', kernel_initializer=initializers.random_normal(stddev=0.03, seed=98765), \
                       kernel_regularizer=regularizers.l2(0.5))
        classifier.add(layer1)
        # Compiling the neural network
        sgd = optimizers.SGD(lr=learning_rate, clipvalue=0.5)
        classifier.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
        return classifier

    def lr_pzi(self,classifier, X_train, X_test, y_train, y_test, steps):
        classifier.fit(X_train, y_train, epochs=steps, verbose=0)
        theta_i = classifier.predict(X_test)
        loss_and_metrics = classifier.evaluate(X_test, y_test)
        print(theta_i[1:10])
        eval_model = accuracy_score(y_test, np.where(theta_i > 0.5, 1, 0))
        print("eval model",eval_model)
        weights = classifier.get_weights()
        return theta_i, eval_model,loss_and_metrics, weights[0]

    def nn_pzi(self,classifier, social_features, y, steps, true_labels):
        classifier.fit(social_features, y, epochs=steps, verbose=0)
        theta_i = classifier.predict(social_features)
        eval_model = accuracy_score(true_labels, np.where(theta_i > theta_i.mean(), 1, 0))
        weights = classifier.get_weights()
        return theta_i, eval_model, weights[0]

    def nn_pzi_test_val(self, classifier, social_features, prob_e_step, steps):
        classifier.fit(social_features, prob_e_step, epochs=steps, verbose=0)
        theta_i = classifier.predict(social_features)
        weights = classifier.get_weights()
        return theta_i, weights[0],classifier

    def train_m_step(self, classifier, social_features, prob_e_step,
                       steps, total_epochs, y_test, y_val,strat_val):
        theta_i = prob_e_step.copy()
        weights = np.array([])
        iter = 0
        old_theta_i = np.zeros((social_features.shape[0], 1))
        epsilon = 1e-3
        while (LA.norm(theta_i - old_theta_i) > epsilon) and (iter < total_epochs):
            # if (iter % 5 == 0) and (iter>0):
            #     min_norm = LA.norm(theta_i - old_theta_i)
            old_theta_i = theta_i.copy()
            theta_i, weights, classifier = self.nn_pzi_test_val(classifier, social_features, prob_e_step, steps)
            end_val = strat_val + y_val.shape[0]
            theta_i_test = theta_i[strat_val:(end_val+1)]
            theta_i_val = theta_i[(end_val+1):]
            eval_model_test = accuracy_score(y_test, np.where(theta_i_test > 0.5, 1, 0))
            eval_model_val = accuracy_score(y_val, np.where(theta_i_val > 0.5, 1, 0))
            if iter%10==0:
                print ("epoch", iter," convergence influencer:", LA.norm(theta_i - old_theta_i),"val", eval_model_val,\
                    "test", eval_model_test)
            iter +=1
        print ("epoch", iter, " convergence influencer:", LA.norm(theta_i - old_theta_i), "val", eval_model_val, \
            "test", eval_model_test)
        return theta_i,classifier, weights

    def train(self,classifier,social_features,true_labels, p_z_i_1, total_epochs, steps, size_train):
        y = np.concatenate((true_labels[0:size_train], p_z_i_1[size_train:]))
        for i in range(total_epochs):
            #print("epoch", i)
            theta_i, eval_model, weights = self.nn_pzi(classifier, social_features, y, steps,true_labels)
            y = np.concatenate((true_labels[0:size_train], theta_i[size_train:]))
            result = pd.DataFrame(data=np.concatenate([np.where(theta_i > theta_i.mean(), 1, 0), true_labels], axis=1),
                                  columns=['classification', 'truth'])
            #print("evaluation", eval_model)
        return result, eval_model,weights, classifier.metrics_names, theta_i,classifier


    def logistic_regression(self,input_file,output_file,true_labels,weights_file,total_epochs,learning_rate):
        simple_example = pd.read_csv(input_file, sep=",")
        social_features = simple_example[['follower_nbr', 'followee_nbr', 'tweets_nbr', 'avg_length_tweets']].values
        #social_features = simple_example.values
        labels = pd.read_csv(true_labels, sep=",")
        true_labels = labels[['label']].values
        X_train, X_test, y_train, y_test = train_test_split(social_features, true_labels, test_size = 0.2, random_state=45)
        n = social_features.shape[0]
        print("n=",n)
        print ("true_labels", true_labels.shape[0])
        m = social_features.shape[1]
        # initi pzi
        p_z_i_0, p_z_i_1 = self.init_probabilities(n)

        n_neurons = 3
        steps = 1
        hidden = False
        size_train = int(0.6 * n)
        classifier = self.define_nn(n_neurons, hidden, m,learning_rate)
        for i in range(total_epochs):
            theta_i, eval_model,loss_and_metrics, weights = self.lr_pzi(classifier, X_train, X_test, y_train, y_test, steps)
            result = pd.DataFrame(data=np.concatenate([np.where(theta_i > 0.5, 1, 0), y_test], axis=1), columns=['classification', 'truth'])
            np.savetxt(weights_file,weights,delimiter=',')
            result.to_csv(output_file)
            metrics = pd.DataFrame(np.array(eval_model).reshape(1, 1), columns=[['accuracy']])
            with open(output_file, 'a') as f:
                metrics.to_csv(f, header=True)

    def nn(self, input_file,output_file,weights_file,total_epochs,learning_rate):
        simple_example = pd.read_csv(input_file, sep=",")
        social_features = simple_example[['follower_nbr', 'followee_nbr', 'tweets_nbr', 'avg_length_tweets']].values
        true_labels = simple_example[['label']].values
        n = social_features.shape[0]
        m = social_features.shape[1]
        # initi pzi
        p_z_i_0, p_z_i_1 = self.init_probabilities(n)

        n_neurons = 3
        steps = 1
        hidden = True
        size_train = int(0.8 * n)
        classifier = self.define_nn(n_neurons, hidden, m,learning_rate)
        result, eval_model, weights, metrics_names, theta_i = self.train(classifier,social_features, true_labels, p_z_i_1, total_epochs,
                                                                steps, size_train)
        np.savetxt(weights_file,weights,delimiter=',')
        result.to_csv(output_file)
        metrics = pd.DataFrame(np.array(eval_model).reshape(1, 1), columns=[['accuracy']])
        with open(output_file, 'a') as f:
            metrics.to_csv(f, header=True)

