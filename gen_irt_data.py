from __future__ import division

import os
import sys
import pandas as pd
import argparse

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis,LinearDiscriminantAnalysis
from sklearn.metrics import *

from models.tree import MyDecisionTreeClassifier
from data.synthetic import get_synthetic_data
from data.load_data import load_mnist_fashion_data
from visualization.plots import *


classifiers = {
    'MLP': MLPClassifier(max_iter=500,hidden_layer_sizes=(256,64,)),##
    #'MLP2': MLPClassifier(max_iter=500,hidden_layer_sizes=(128,32,)),
    #'RBF SVM': SVC(probability=True),
    #'Linear SVM': SVC(kernel='linear', probability=True),
    'Logistic Regression': LogisticRegression(),##
    'Decision Tree': MyDecisionTreeClassifier(),##
    'Nearest Neighbors': KNeighborsClassifier(3),##
    #'Gaussian process': GaussianProcessClassifier(1.0 * RBF(1.0)),
    'Random Forest': RandomForestClassifier(),
    'Adaboost': AdaBoostClassifier(),
    'Naive Bayes': GaussianNB(),#
    'LDA': LinearDiscriminantAnalysis(),
    'QDA': QuadraticDiscriminantAnalysis(),
    #'Weak Classifier': 'weak',
    #'Strong Classifier': 'strong',
    'Positive Classifier': 'positive',
    'Negative Classifier': 'negative',
    'Constant Classifier': 'uncertain',##
    #'Perturb-MLP': 'perturb-MLP',
    #'Perturb-LR': 'perturb-Logistic Regression'
}


def weak_baseline(y_test):
    probas = np.ones(len(y_test)).reshape(-1, 1)
    probas[y_test == 1] = 0.0
    probas = np.hstack((1.0 - probas, probas))
    predictions = np.logical_not(y_test)
    return probas, predictions.astype(int)

def strong_baseline(y_test):
    probas = np.zeros(len(y_test)).reshape(-1, 1)
    probas[y_test == 1] = 1.0
    probas = np.hstack((1.0 - probas, probas))
    predictions = np.logical_not(y_test)
    return probas, predictions.astype(int)


def positive_baseline(y_test):
    probas = np.ones(len(y_test)).reshape(-1, 1)
    predictions = np.ones(len(y_test))
    probas = np.hstack((1.0 - probas, probas))
    return probas, predictions.astype(int)

def negative_baseline(y_test):
    probas = np.zeros(len(y_test)).reshape(-1, 1)
    predictions = np.zeros(len(y_test))
    probas = np.hstack((1.0 - probas, probas))
    return probas, predictions.astype(int)


def uncertain_baseline(y_test):
    probas = 0.5 * np.ones(len(y_test)).reshape(-1, 1)
    #probas = np.random.normal(loc=0.5,scale=0.1,size=len(y_test)).reshape(-1,1)
    probas = np.hstack((1.0 - probas, probas))
    #predictions = np.ones(len(y_test))
    #predictions = np.argmax(probas,axis=1)
    predictions = np.random.choice(2,len(y_test))
    #print(predictions.shape)
    return probas, predictions.astype(int)


def get_probas_and_predictions(classifier, X_test, y_test, pclassifier=None):
    if classifier == 'weak':
        return weak_baseline(y_test)
    if classifier == 'strong':
        return strong_baseline(y_test)
    if classifier == 'positive':
        return positive_baseline(y_test)
    if classifier == 'negative':
        return negative_baseline(y_test)
    if classifier == 'uncertain':
        return uncertain_baseline(y_test)
    if isinstance(classifier,str) and 'perturb' in classifier:
        print(classifier)
        probas = pclassifier.predict_proba(X_test)
        #predictions = pclassifier.predict(X_test).astype(int)
        n = len(y_test)
        pidx = np.random.choice(n, int(0.4 * n), replace=False)
        probas[pidx,0] = np.random.uniform(0.,1.,size=len(pidx))
        probas[pidx,1] = 1 - probas[pidx,0]
        predictions = np.argmax(probas,axis=1)
        #print(probas.shape,predictions.shape)
        return probas,predictions
        
    probas = classifier.predict_proba(X_test)
    predictions = classifier.predict(X_test).astype(int)

    return probas, predictions


def evaluate(classifier, X_test, y_test,recal,pclassifier=None):
    if isinstance(classifier,str) and 'perturb' in classifier:
        probas, predictions = get_probas_and_predictions(classifier, X_test, y_test, pclassifier)
    else:
        probas, predictions = get_probas_and_predictions(classifier, X_test, y_test)
    incorrect = predictions != y_test
    correct = predictions == y_test

    if recal:
        lr = LogisticRegression()
        lr.fit(probas,predictions)
        probas = lr.predict_proba(probas)
 

    chance = probas[np.arange(len(probas)), predictions]
    chance[incorrect] = 1.0 - chance[incorrect]

    return probas[:, 1], chance, np.array([[np.sum(correct) + 1, np.sum(
     incorrect) + 1]]), correct, incorrect


def cl_info(classifier, X_train, y_train, X_test, y_test,recal=False):
    pclassifier = None
    if not isinstance(classifier,str):
        classifier.fit(X_train, y_train)
    elif 'perturb' in classifier:
        pclassifier = classifiers[classifier.split('-')[1]]
        pclassifier.fit(X_train, y_train)
    p, p_irt, prior, correct, incorrect = evaluate(classifier, X_test, y_test,recal,pclassifier)
    return p, p_irt, prior, correct, incorrect


def gen_IRT_data(dataset_type,size,nfrac,seed,results_path,cl1,cl2,train_noise=True,recal=False):
    
    if not dataset_type in ['moons','clusters','circles','fashion','mnist']:
        raise TypeError('Non-supported dataset type!')

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    np.random.seed(seed)
    if dataset_type in ['fashion','mnist']:
        #print(nfrac,(100*nfrac))
        spec_name = dataset_type+'_s'+str(size)+'_f'+str(int(100*nfrac))+'_sd'+str(seed)+'_cl'+str(cl1)+str(cl2)
    else:
        spec_name = dataset_type+'_s'+str(size)+'_f'+str(int(100*nfrac))+'_sd'+str(seed)

    if recal:
        spec_name += '_cal'
    spec_name+='_m'+str(len(classifiers))

    
    if train_noise:
        tnfrac = nfrac
    else:
        tnfrac = 0.
    if dataset_type == 'fashion':
        X,y,X_test,y_test,nidx = load_mnist_fashion_data(train_size=1000,test_size=size,\
                                    dpath='../data/fashion',noise_frac=nfrac,train_noise_frac=tnfrac,\
                                    class1=cl1, class2=cl2,seed=seed)
    elif dataset_type == 'mnist':
        X,y,X_test,y_test,nidx = load_mnist_fashion_data(train_size=1000,test_size=size,\
                                    dpath='../data/mnist',noise_frac=nfrac,train_noise_frac=tnfrac,\
                                    class1=cl1,class2=cl2,seed=seed)
    else:
        #print(tnfrac)
        X, y, _ = get_synthetic_data(dataset_type, size,noise_frac=tnfrac)

        X_test, y_test, nidx = get_synthetic_data(dataset_type, size, noise_frac=nfrac)


    correct = np.ones(len(y_test))
    incorrect = np.ones(len(y_test))
    probas = []
    probas_irt = []
    cl_priors = []
    labels = []
    accuracy = []
    f1 = []
    for classifier in classifiers:
        labels.append(classifier)
        cl = classifiers[classifier]
        p, p_irt, prior, c, i = cl_info(cl, X, y, X_test, y_test,recal)
        accuracy.append(c.sum()*1./len(y_test))
        pred = np.zeros_like(c)
        pred[c] = y_test[c]
        pred[i] = 1 - y_test[i]
        f1.append(f1_score(y_test,pred))
        correct += c
        incorrect += i
        probas.append(p)
        probas_irt.append(p_irt)
        cl_priors.append(prior)

    if dataset_type in ['moons','clusters']:
        fig = plot_probabilities(X_test, [y_test], ['labels'], 'original')
        fig.savefig(os.path.join(results_path, 'labels_{}.pdf'.format(spec_name)))

    probas = np.vstack(probas)
    probas_irt = np.vstack(probas_irt)
    probas_irt = np.clip(probas_irt, 1e-3, 1. - 1e-3)

    it_priors = np.hstack((incorrect.reshape(-1, 1), correct.reshape(-1, 1)))
    cl_priors = np.vstack(cl_priors)

    sdata = pd.DataFrame()
    pdata = pd.DataFrame()
    acu = pd.DataFrame()
    acu['classifier'] = classifiers.keys()
    acu['accuracy'] = 0.
    acu['f1_score'] = 0.
    for i,cname in enumerate(classifiers.keys()):
        print(i,cname,accuracy[i])
        sdata[cname] = probas_irt[i]
        pdata[cname] = probas[i]
        acu.loc[i,'accuracy'] = accuracy[i]
        acu.loc[i,'f1_score'] = f1[i]
        print('average response:',np.mean(probas_irt[i]))
       

    sdata.to_csv('irt_data_'+spec_name+'.csv',index=False)
    pdata.to_csv('predict_data_prob_'+spec_name+'.csv',index=False)
    acu.to_csv(results_path+'/model_acu_'+spec_name+'.csv',index=False)
    
    xtest = pd.DataFrame(index=range(size))
    if not dataset_type in ['fashion','mnist']:
        xtest['x'] = X_test[:,0]
        xtest['y'] = X_test[:,1]
    xtest['noise'] = 0
    xtest['label'] = y_test
    if not nidx is None:
        xtest.loc[nidx,'noise'] = 1
    print(xtest.shape)
    #print(nidx)
    xtest.to_csv('xtest_'+spec_name+'.csv',index=False)

def str2bool(x):
    if x.lower() == 'false':
        return False
    else:
        return True



parser = argparse.ArgumentParser()
parser.add_argument('-ds','--dataset', default='moons', type=str, help='can be moons, clusters, mnist,fashion')
parser.add_argument('-rp','--result_path', default='./results/', type=str, help='result path')
parser.add_argument('-dsz','--data_size', default=400, type=int, help='number of data points in each class')
parser.add_argument('-nf','--noise_fraction', default=0.2, type=float, help='noise fraction')
parser.add_argument('-cl1','--class1', default=7, type=int, help='specify first class in mnist or fashion mnist')
parser.add_argument('-cl2','--class2', default=9, type=int, help='specify second class in mnist or fashion mnist')
parser.add_argument('-tn','--train_noise', default=False, type=str2bool, help='if inject noise in training set, set to True')
parser.add_argument('-sd','--seed', default=42, type=int, help='random seed')

args = parser.parse_args()
print('seed',args.seed)
np.random.seed(args.seed)


gen_IRT_data(args.dataset, args.data_size, args.noise_fraction, args.seed, \
            args.result_path+args.dataset,args.class1,args.class2,args.train_noise)


