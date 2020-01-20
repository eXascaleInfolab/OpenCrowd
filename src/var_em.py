import pandas as pd
import csv
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense
from math import floor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from scipy.special import digamma
from sklearn.preprocessing import StandardScaler
from numpy import linalg as LA
from nn_em import nn_em
import random
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
from sklearn import metrics
import argparse

def init_probabilities(n_infls):
    # initialize probability z_i (item's quality) randomly
    qz1 = 0.5 * np.ones((n_infls, 1))
    # initialize probability alpha beta (worker's reliability)
    A = 2
    B = 2
    return 1 - qz1, qz1, A, B


def init_alpha_beta(A, B, n_workers):
    alpha = np.zeros((n_workers, 1),dtype='float32')
    beta = np.zeros((n_workers, 1),dtype='float32')
    for w in range(0, n_workers):
        alpha[w] = A
        beta[w] = B
    return alpha, beta


def update(a, b,n_update,change):
    n_update += 1
    change += np.abs(a - b)
    return n_update,change

def optimize_rj(x_train, n_neurons, nb_layers, training_epochs, display_step, batch_size, n_input, alpha, beta):
    graph1 = tf.Graph()
    with graph1.as_default():
        tf.set_random_seed(1)
        # input layer
        x = tf.placeholder(tf.float32, [None, n_input])
        keep_prob = tf.placeholder(tf.float32)
        layer = x
        # hideen layers
        for _ in range(nb_layers):
            layer = tf.layers.dense(inputs=layer, units=n_neurons, activation=tf.nn.tanh)
        # output layer
        alpha_prime = tf.layers.dense(inputs=layer, units=1, activation=lambda x: tf.nn.relu(x) + 1)
        beta_prime = tf.layers.dense(inputs=layer, units=1, activation=lambda x: tf.nn.relu(x) + 1)

        dist = tf.distributions.Beta(alpha_prime, beta_prime)
        target_dist = tf.distributions.Beta(alpha, beta)

        loss = tf.distributions.kl_divergence(target_dist, dist)
        cost = tf.reduce_mean(loss)

        optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

    with tf.Session(graph=graph1) as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(training_epochs):
            avg_cost = 0.0
            total_batch = int(len(x_train) / batch_size)
            x_batches = np.array_split(x_train, total_batch)
            for i in range(total_batch):
                batch_x = x_batches[i]
                _, c = sess.run([optimizer, cost],
                                feed_dict={
                                    x: batch_x,
                                    keep_prob: 0.8
                                })
                avg_cost += c / total_batch
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", \
                      "{:.9f}".format(avg_cost))
        print("Optimization Finished!")

        alpha_prime_res, beta_prime_res = sess.run([alpha_prime, beta_prime],
                                                   feed_dict={
                                                       x: x_train,
                                                       keep_prob: 0.8
                                                   })
        print("alpha_prime_res=", alpha_prime_res, "beta_prime_res=", beta_prime_res)
        return alpha_prime_res, beta_prime_res


def e_step(n_infls_unlabel,y_train, n_workers, q_z_i_0, q_z_i_1, annotation_matrix, alpha, beta, theta_i,true_labels,new_order, max_it=100):
    for it in range(max_it):
        change = 0
        n_update = 0
        # update q(z)
        #all_workers = np.unique(annotation_matrix[:, 0])
        for infl in new_order.tolist():
            index_infl = np.where(new_order == infl)[0][0]
            updated_q_z_i_0 = (1 - theta_i[index_infl])
            updated_q_z_i_1 = theta_i[index_infl]
            infl_aij = annotation_matrix[annotation_matrix[:, 1] == infl]
            T_i = infl_aij[infl_aij[:, 2] == 1][:, 0]
            for worker in T_i.astype(int):
                #worker_id = np.where(all_workers == worker)
                alpha_val = alpha[worker]
                beta_val =  beta[worker]
                updated_q_z_i_0 = updated_q_z_i_0 * np.exp(digamma(beta_val) - digamma(alpha_val + beta_val))
                updated_q_z_i_1 = updated_q_z_i_1 * np.exp(digamma(alpha_val) - digamma(alpha_val + beta_val))

            T_i_n_all = infl_aij[infl_aij[:, 2] == 0][:, 0]
            for worker in T_i_n_all.astype(int):
                #worker_id = np.where(all_workers == worker)
                alpha_val = alpha[worker]
                beta_val =  beta[worker]
                updated_q_z_i_0 = updated_q_z_i_0 * np.exp(digamma(alpha_val) - digamma(alpha_val + beta_val))
                updated_q_z_i_1 = updated_q_z_i_1 * np.exp(digamma(beta_val) - digamma(alpha_val + beta_val))

            new_q_z_i_1 = updated_q_z_i_1 * 1.0 / (updated_q_z_i_0 + updated_q_z_i_1)
            n_update, change = update(q_z_i_1[index_infl], new_q_z_i_1,n_update,change)
            q_z_i_0[index_infl] = updated_q_z_i_0 * 1.0 / (updated_q_z_i_0 + updated_q_z_i_1)
            q_z_i_1[index_infl] = updated_q_z_i_1 * 1.0 / (updated_q_z_i_0 + updated_q_z_i_1)

        q_z_i_1_ = np.concatenate((q_z_i_1[:n_infls_unlabel], y_train, q_z_i_1[n_infls_unlabel + y_train.shape[0]:]))
        q_z_i_0_ = np.concatenate((q_z_i_0[:n_infls_unlabel], 1 - y_train, q_z_i_0[n_infls_unlabel + y_train.shape[0]:]))

        # update q(r)
        new_alpha = np.zeros((n_workers, 1))
        new_beta = np.zeros((n_workers, 1))
        for worker in range(0, n_workers):
            new_alpha[worker] = alpha[worker]
            new_beta[worker] = beta[worker]

        for worker in range(0, n_workers):
            worker_aij = annotation_matrix[annotation_matrix[:, 0] == worker]
            T_j_1 = worker_aij[worker_aij[:,2] == 1][:, 1]
            for infl in T_j_1.astype(int):
                if (np.where(new_order == infl)[0].shape[0]) > 0:
                    index_infl = np.where(new_order == infl)[0][0]
                    new_alpha[worker] += q_z_i_1_[index_infl]
                    new_beta[worker] += 1 - q_z_i_1_[index_infl]
                #print worker,infl,1,theta_i[infl]
            T_j_0 = worker_aij[worker_aij[:, 2] == 0][:, 1]
            for infl in T_j_0.astype(int):
                if (np.where(new_order == infl)[0].shape[0]) > 0:
                    index_infl = np.where(new_order == infl)[0][0]
                    new_alpha[worker] += (1 - q_z_i_1_[index_infl])
                    new_beta[worker] += q_z_i_1_[index_infl]
                #print worker,infl,0,theta_i[infl]

        for worker in range(0, n_workers):
            n_update, change = update(alpha[worker], new_alpha[worker],n_update,change)
            alpha[worker] = new_alpha[worker]
            n_update, change = update(beta[worker], new_beta[worker],n_update,change)
            beta[worker] = new_beta[worker]
        avg_change = change * 1.0 / n_update
        if avg_change < 0.01:
            break
        #print "qz1", q_z_i_1
        #print "alpha", alpha
        #print "beta", beta
        return q_z_i_0_,q_z_i_1_,alpha,beta

def m_step(nn_em,q_z_i_0,q_z_i_1, classifier, social_features, total_epochs, steps, y_test, y_val,strat_val,alpha, beta):
    #print prob_e_step
    prob_e_step = np.where(q_z_i_0 > 0.5, 0, 1)
    theta_i, classifier, weights = nn_em.train_m_step(classifier, social_features,
                                                      q_z_i_1,
                                                      steps, total_epochs, y_test, y_val,strat_val)
    # n_neurons = 3
    # nb_layers = 0
    # training_epochs = 20
    # display_step = 10
    # batch_size = 1
    # n_input = worker_data.shape[1]
    # alpha_prime_res, beta_prime_res = optimize_rj(worker_data, n_neurons, nb_layers, training_epochs, display_step,
    #                                               batch_size, n_input, alpha, beta)
    # alpha = alpha_prime_res
    # beta = beta_prime_res
    return theta_i,classifier


def var_em(nn_em_in,n_infls_unlabel, n_infls_label,aij_s,new_order, n_workers, social_features_labeled,social_features_unlabeled, true_labels, supervision_rate, \
           column_names, n_neurons, hidden, m_feats, weights_before_em,weights_after_em,iterr,total_epochs,evaluation_file,theta_file,steps,nb_hidden_layer):
    n_infls = n_infls_unlabel + n_infls_label
    q_z_i_0, q_z_i_1, A, B = init_probabilities(n_infls)
    alpha, beta = init_alpha_beta(A, B, n_workers)

    X_train, X_test, y_train, y_test = train_test_split(social_features_labeled, true_labels,
                                                        test_size=(1 - supervision_rate), shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, shuffle=False)
    classifier = nn_em_in.define_nn(n_neurons, hidden, m_feats, nb_hidden_layer, 0.001)
    steps_it0 = 0
    epsilon = 1e-4
    theta_i = q_z_i_1.copy()
    old_theta_i = np.zeros((n_infls, 1))

    while (LA.norm(theta_i - old_theta_i) > epsilon) and (steps_it0 < total_epochs):
        classifier.fit(X_train, y_train, epochs=steps, verbose=0)
        theta_i_val = classifier.predict(X_val)
        theta_i_test = classifier.predict(X_test)
        theta_i_unlabeled = classifier.predict(social_features_unlabeled)
        theta_i = np.concatenate((theta_i_unlabeled, y_train, theta_i_val, theta_i_test))
        eval_model_test = accuracy_score(y_test, np.where(theta_i_test > 0.5, 1, 0))
        eval_model_val = accuracy_score(y_val, np.where(theta_i_val > 0.5, 1, 0))
        if steps_it0 % 10 == 0:
            print "epoch", steps_it0, " convergence:", LA.norm(theta_i - old_theta_i), \
                "val", eval_model_val, "test", eval_model_test
        steps_it0 += 1

    weights = classifier.get_weights()
    pd.DataFrame(np.concatenate((column_names[1:], weights[0]), axis=1)).to_csv(weights_before_em, encoding="utf-8")

    eval_model_test = accuracy_score(y_test, np.where(theta_i_test > 0.5, 1, 0))
    eval_model_val = accuracy_score(y_val, np.where(theta_i_val > 0.5, 1, 0))
    strat_val = n_infls_unlabel + X_train.shape[0]
    end_val = n_infls_unlabel + X_train.shape[0] + X_val.shape[0]

    auc_val = roc_auc_score(y_val, theta_i_val)
    auc_test = roc_auc_score(y_test, theta_i_test)
    #auc_test = 0
    precision_val_theta, recall_val_theta, thresholds_val_theta = precision_recall_curve(y_val,
                                                                                         theta_i_val)
    precision_test_theta, recall_test_theta, thresholds_test_theta = precision_recall_curve(y_test, theta_i_test)

    auprc_val_theta = metrics.auc(precision_val_theta, recall_val_theta, reorder=True)
    auprc_test_theta = metrics.auc(precision_test_theta, recall_test_theta, reorder=True)

    scores_val_theta = precision_recall_fscore_support(y_val, np.where(theta_i_val > 0.5, 1, 0))
    scores_test_theta = precision_recall_fscore_support(y_test, np.where(theta_i_test > 0.5, 1, 0))
    head = "nb iteration,accuracy validation,accuracy test,auc val,auc test,auprc val, auprc test,precision val 0,precision val 1,\
    precision test 0,precision test 1,recall val 0, recall val 1,recall test 0, recall test 1,\
    F1 val 0, F1 val 1,F1 test 0, F1 test 1,accuracy validation_theta,accuracy test_theta,auc val_theta,auc test_theta,\
    auprc val_theta, auprc test_theta,precision val_theta 0,precision val_theta 1, \
    precision test_theta 0,precision test_theta 1,recall val_theta 0, recall val_theta 1,recall test_theta 0, recall test_theta 1,\
    F1 val_theta 0, F1 val_theta 1,F1 test_theta 0, F1 test_theta 1"
    scores = str(-1) + ',0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,' + str(eval_model_val) + ',' + str(
        eval_model_test) + ',' + str(auc_val) + ',' + str(auc_test) + ',' + \
             str(auprc_val_theta) + ',' + str(auprc_test_theta) + ',' + str(scores_val_theta[0][0]) + ',' + str(
        scores_val_theta[0][1]) + ',' + \
             str(scores_test_theta[0][0]) + ',' + str(scores_test_theta[0][1]) + ',' + str(
        scores_val_theta[1][0]) + ',' + str(scores_val_theta[1][1]) + ',' + \
             str(scores_test_theta[1][0]) + ',' + str(scores_test_theta[1][1]) + ',' + str(
        scores_val_theta[2][0]) + ',' + str(scores_val_theta[2][1]) + ',' + \
             str(scores_test_theta[2][0]) + ',' + str(scores_test_theta[2][1])
    print scores
    with open(evaluation_file, 'a') as file:
        file.write("supervision rate," + str(supervision_rate))
        file.write('\n')
        file.write(head)
        file.write('\n')
        file.write(scores)
        file.write('\n')
    theta_i = np.concatenate((theta_i_unlabeled, y_train, theta_i_val, theta_i_test))
    theta_quality = np.concatenate((true_labels, theta_i[social_features_unlabeled.shape[0]:]), axis=1)
    pd.DataFrame(theta_quality).to_csv(theta_file, index=False, header=['Truth', 'quality'])
    social_features = np.concatenate((social_features_unlabeled, social_features_labeled))

    em_step = 0
    while em_step < iterr:
        # variational E step
        q_z_i_0, q_z_i_1, alpha, beta = e_step(n_infls_unlabel, y_train, n_workers, q_z_i_0, q_z_i_1, aij_s, alpha,
                                               beta, theta_i, true_labels,new_order)
        # variational M step
        theta_i, classifier = m_step(nn_em_in, q_z_i_0,q_z_i_1, classifier, social_features, total_epochs, steps, y_test, y_val,
                                     strat_val, alpha, beta)
        em_step += 1
        eval_model_val = accuracy_score(y_val, np.where(q_z_i_0[strat_val:end_val] > 0.5, 0, 1))
        eval_model_test = accuracy_score(y_test, np.where(q_z_i_0[end_val:] > 0.5, 0, 1))
        auc_val = roc_auc_score(y_val, q_z_i_1[strat_val:end_val])
        auc_test = roc_auc_score(y_test, q_z_i_1[end_val:])
        #auc_test = 0
        eval_model_val_theta = accuracy_score(y_val, np.where(theta_i[strat_val:end_val] > 0.5, 1, 0))
        eval_model_test_theta = accuracy_score(y_test, np.where(theta_i[end_val:] > 0.5, 1, 0))
        auc_val_theta = roc_auc_score(y_val, theta_i[strat_val:end_val])
        auc_test_theta = roc_auc_score(y_test, theta_i[end_val:])
        #auc_test_theta = 0

        precision_val, recall_val, thresholds_val = precision_recall_curve(y_val, q_z_i_1[strat_val:end_val])
        precision_test, recall_test, thresholds_test = precision_recall_curve(y_test, q_z_i_1[end_val:])

        auprc_val = metrics.auc(precision_val, recall_val,reorder=True)
        auprc_test = metrics.auc(precision_test, recall_test,reorder=True)

        scores_val = precision_recall_fscore_support(y_val, np.where(q_z_i_1[strat_val:end_val] > 0.5, 1, 0),labels=[0,1])
        scores_test = precision_recall_fscore_support(y_test, np.where(q_z_i_1[end_val:] > 0.5, 1, 0),labels=[0,1])

        precision_val_theta, recall_val_theta, thresholds_val_theta = precision_recall_curve(y_val, theta_i[strat_val:end_val])
        precision_test_theta, recall_test_theta, thresholds_test_theta = precision_recall_curve(y_test, theta_i[end_val:])

        auprc_val_theta = metrics.auc(precision_val_theta, recall_val_theta,reorder=True)
        auprc_test_theta = metrics.auc(precision_test_theta, recall_test_theta,reorder=True)

        scores_val_theta = precision_recall_fscore_support(y_val, np.where(theta_i[strat_val:end_val] > 0.5, 1, 0),labels=[0,1])
        scores_test_theta = precision_recall_fscore_support(y_test, np.where(theta_i[end_val:] > 0.5, 1, 0),labels=[0,1])

        print "\n\n"
        scores= str(em_step)+','+ str(eval_model_val) +','+str(eval_model_test)+','+str(auc_val) +','+ str(auc_test)+','+\
                str(auprc_val)+','+str(auprc_test)+','+str(scores_val[0][0])+','+str(scores_val[0][1])+','+ \
                str(scores_test[0][0])+','+str(scores_test[0][1])+','+str(scores_val[1][0])+','+str(scores_val[1][1])+','+\
                str(scores_test[1][0])+','+str(scores_test[1][1])+','+str(scores_val[2][0])+','+ str(scores_val[2][1])+','+ \
                str(scores_test[2][0])+','+str(scores_test[2][1])+','+str(eval_model_val_theta) +','+str(eval_model_test_theta)+','+\
                str(auc_val_theta) +','+ str(auc_test_theta)+','+\
                str(auprc_val_theta)+','+str(auprc_test_theta)+','+str(scores_val_theta[0][0])+','+str(scores_val_theta[0][1])+','+\
                str(scores_test_theta[0][0])+','+str(scores_test_theta[0][1])+','+str(scores_val_theta[1][0])+','+str(scores_val_theta[1][1])+','+\
                str(scores_test_theta[1][0])+','+str(scores_test_theta[1][1])+','+str(scores_val_theta[2][0])+','+ str(scores_val_theta[2][1])+','+ \
                str(scores_test_theta[2][0])+','+str(scores_test_theta[2][1])
        print scores
        with open(evaluation_file, 'a') as file:
            file.write(scores)
            file.write('\n')

    weights = classifier.get_weights()
    pd.DataFrame(np.concatenate((column_names[1:], weights[0]), axis=1)).to_csv(weights_after_em, encoding="utf-8")
    return q_z_i_0, q_z_i_1, alpha, beta, theta_i, classifier

def parse_args():
    parser = argparse.ArgumentParser(
        description="EM method")
    parser.add_argument("--labeled_social_features",
                        type=str,
                        required=True,
                        help="inputfile labeled social features")

    parser.add_argument("--unlabeled_social_features",
                        type=str,
                        required=True,
                        help="inputfile unlabeled social features")

    parser.add_argument("--annotation_matrix",
                        type=str,
                        required=True,
                        help="inputfile of the annotation matrix")

    parser.add_argument("--labels",
                        type=str,
                        required=True,
                        help="inputfile of labels")

    parser.add_argument("--total_epochs_nn",
                        default=10,
                        type=int,
                        help="number of epochs for the Neural network at the M step")

    parser.add_argument("--total_neurons_nn",
                        default=10,
                        type=int,
                        help="number of neurons for the Neural network at the M step")

    parser.add_argument("--nb_hidden_layer",
                        default=1,
                        type=int,
                        help="number of hidden layer for the Neural network at the M step")

    parser.add_argument("--steps",
                        default=1,
                        type=int,
                        help="number of steps for the Neural network at the M step")

    parser.add_argument("--hidden_layer",
                        default=False,
                        type=bool,
                        help="use hidden layer in the NN")

    parser.add_argument("--supervision_rate",
                        default=0.6,
                        type=float,
                        help="how much to use for training")

    parser.add_argument("--sampling_rate",
                        default=1.0,
                        type=float,
                        help="how much to use for negative sampling for the e step")

    parser.add_argument("--nb_iterations_EM",
                        default=10,
                        type=int,
                        help="number of iterations for the EM")

    parser.add_argument("--worker_reliability_file",
                        type=str,
                        required=True,
                        help="worker reliability file output of the model")

    parser.add_argument("--influencer_quality_file",
                        type=str,
                        required=True,
                        help="influencer quality file output of the model")

    parser.add_argument("--evaluation_file",
                        type=str,
                        required=True,
                        help="evaluation result after each iteration")

    parser.add_argument("--theta_file",
                        type=str,
                        required=True,
                        help="theta result after nn")

    parser.add_argument("--weights_before_em",
                        type=str,
                        required=True,
                        help="inputfile LR")

    parser.add_argument("--weights_after_em",
                        type=str,
                        required=True,
                        help="inputfile weights EM")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    influencer_file_labeled = args.labeled_social_features #'../input/simple_example_vem_labeled.csv'
    influencer_file_unlabeled = args.unlabeled_social_features # '../input/simple_example_vem_unlabeled.csv'
    annotation_file = args.annotation_matrix #'../input/aij_simple_example_vem.csv'
    labels_file = args.labels #'../input/simple_example_vem_labels.csv'
    theta_file = args.theta_file # '../output/theta_i_vem_se.csv'
    evaluation_file = args.evaluation_file # '../output/evaluation_vem_se.csv'
    weights_before_em = args.weights_before_em #'../output/weights_before_em.csv'
    weights_after_em = args.weights_after_em #'../output/weights_after_em.csv'
    nb_hidden_layer = args.nb_hidden_layer


    influencer_labeled = pd.read_csv(influencer_file_labeled, sep=",").drop(['language', 'user_name'], axis=1)
    influencer_unlabeled = pd.read_csv(influencer_file_unlabeled, sep=",").drop(['language', 'user_name'], axis=1)

    column_names = np.array(influencer_labeled.columns).reshape((influencer_labeled.shape[1], 1))
    print column_names.shape
    annotation_matrix = np.loadtxt(annotation_file, delimiter=',')
    labels = pd.read_csv(labels_file, sep=",")

    social_features_labeled = preprocessing.scale(influencer_labeled.values[:, 1:])
    true_labels_pr = labels[['label']].values

    print influencer_labeled.values[:, [0]].shape,social_features_labeled.shape,true_labels_pr.shape

    social_features_labeled = np.concatenate(
        (influencer_labeled.values[:, [0]], social_features_labeled, true_labels_pr), axis=1)
    soc_label_bsh = social_features_labeled.copy()
    #np.random.shuffle(social_features_labeled)

    social_features_unlabeled = preprocessing.scale(influencer_unlabeled.values[:, 1:])
    social_features_unlabeled = np.concatenate((influencer_unlabeled.values[:, [0]], social_features_unlabeled), axis=1)
    soc_unlabel_bsh = social_features_unlabeled.copy()
    #np.random.shuffle(social_features_unlabeled)

    m = social_features_labeled.shape[1]
    true_labels = social_features_labeled[:, [(m - 1)]]
    social_features_labeled = social_features_labeled[:, :(m - 1)]

    n_infls_label = social_features_labeled.shape[0]
    n_infls_unlabel = social_features_unlabeled.shape[0]
    m_feats = social_features_labeled.shape[1]
    n_workers = np.unique(annotation_matrix[:, 0]).shape[0]

    new_order = np.concatenate((social_features_unlabeled[:, 0], social_features_labeled[:, 0]), axis=0)

    total_epochs = args.total_epochs_nn#10
    n_neurons = args.total_neurons_nn #3
    hidden = args.hidden_layer #false
    steps = args.steps #1
    supervision_rate = args.supervision_rate #0.6
    iterr = args.nb_iterations_EM #10
    sampling_rate = args.sampling_rate #2.0

    aij = np.empty((0, 3), int)
    for worker in range(0, n_workers):
        worker_aij = annotation_matrix[annotation_matrix[:, 0] == worker]
        worker_aij_s = worker_aij.copy()
        for i in range(0, n_infls_label + n_infls_unlabel):
            worker_aij_s[i, :] = worker_aij[worker_aij[:, 1] == new_order[i]]
        aij = np.concatenate((aij, worker_aij_s))
    all_workers = np.unique(annotation_matrix[:, 0])
    aij_s = np.empty((0, 3), int)

    for worker in all_workers:
        worker_aij = annotation_matrix[annotation_matrix[:, 0] == worker]
        T_w = worker_aij[worker_aij[:, 2] == 1]
        T_w_n_all = worker_aij[worker_aij[:, 2] == 0]
        if int(T_w.shape[0] * sampling_rate) < T_w_n_all.shape[0]:
            indices = random.sample(range(T_w_n_all.shape[0]), int(T_w.shape[0] * sampling_rate))
        else:
            indices = random.sample(range(T_w_n_all.shape[0]), T_w_n_all.shape[0])
        T_w_n = T_w_n_all[indices, :]
        aij_s = np.concatenate((aij_s, T_w, T_w_n))

    size_train = int(supervision_rate * n_infls_label)
    percentage_train = 0
    for infl in range(size_train):
        infl_idx = social_features_labeled[infl, 0]
        infl_aij = annotation_matrix[annotation_matrix[:, 1] == infl_idx]
        percentage_train += np.sum(infl_aij[:, 2])

    print "% of ones in the training=", (percentage_train * 100) / aij_s.shape[0]

    print np.sum(aij_s[:, 2]), aij_s.shape[0]
    print "% of ones in the matrix=", (np.sum(aij_s[:, 2]) * 100) / aij_s.shape[0]


    with open(evaluation_file, 'a') as file:
        file.write("sampling rate," + str(sampling_rate))
        file.write('\n')
        file.write("hidden," + str(hidden))
        file.write('\n')
        file.write("nb layers," + str(nb_hidden_layer))
        file.write('\n')
        file.write("nb neurons," + str(n_neurons))
        file.write('\n')
    nn_em_in = nn_em()
    print social_features_labeled.shape, social_features_unlabeled.shape, true_labels.shape

    social_features_labeled = social_features_labeled[:,1:]
    social_features_unlabeled = social_features_unlabeled[:, 1:]
    m_feats = m_feats - 1
    q_z_i_0, q_z_i_1, alpha, beta, theta_i, classifier = var_em(nn_em_in,n_infls_unlabel,n_infls_label,aij_s,new_order,n_workers,\
                                                                social_features_labeled,social_features_unlabeled,\
                                                                true_labels,supervision_rate, column_names,\
                                                                n_neurons,hidden,m_feats,weights_before_em,weights_after_em,\
                                                                iterr,total_epochs,evaluation_file,theta_file,steps,nb_hidden_layer)
    df = pd.read_csv(weights_before_em,
                     names=['name', 'weight']).sort_values(by=['weight'],ascending=False)
    df.to_csv(weights_before_em)
    df = pd.read_csv(weights_after_em,
                     names=['name', 'weight']).sort_values(by=['weight'],ascending=False)
    df.to_csv(weights_after_em)
    worker_reliability_file = args.worker_reliability_file
    influencer_quality_file = args.influencer_quality_file
    worker_reliability = np.concatenate((np.arange(n_workers).reshape(n_workers, 1), alpha, beta), axis=1)
    influencer_quality = np.concatenate(
        (social_features_labeled[:, [0]], true_labels, q_z_i_1[social_features_unlabeled.shape[0]:], theta_i[social_features_unlabeled.shape[0]:]), axis=1)
    pd.DataFrame(worker_reliability).to_csv(worker_reliability_file, index=False, header=['worker', 'alpha','beta'])
    pd.DataFrame(influencer_quality).to_csv(influencer_quality_file, index=False, header=['order', 'Truth', 'quality','theta'])
        # print(pd.DataFrame(data=np.concatenate([np.where(q_z_i_0 > q_z_i_0.mean(), 0, 1), true_labels], axis=1),
        #                    columns=['classification', 'truth']))
# Execute main() function