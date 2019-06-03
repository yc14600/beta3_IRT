from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

import seaborn as sns
import pandas as pd
import glob
import re
from itertools import combinations

import matplotlib
matplotlib.rcParams['text.usetex'] = True



def plot_probabilities(X, probabilities, titles, suptitle):
    norm = plt.Normalize(0, 1)
    n = len(titles)

    nrows = int(np.ceil(n / 2))

    sns.set_context('paper')
    cmap = sns.cubehelix_palette(rot=-.5,light=1.5,dark=-.5,as_cmap=True)

    f, axarr = plt.subplots(nrows, min(n,2))
    if n < 2:
        axarr.scatter(X[:, 0], X[:, 1], c=probabilities[0],
                            cmap=cmap, norm=norm, edgecolor='k',s=60)
        axarr.set_title(titles[0])
        #f.set_size_inches(8, 8)
    else:

        i = j = 0
        for idx, t in enumerate(titles):
            axarr[i, j].scatter(X[:, 0], X[:, 1], c=probabilities[idx],
                                cmap=cmap, norm=norm, edgecolor='k')
            axarr[i, j].set_title(t)
            j += 1
            if j == 2:
                j = 0
                i += 1
        if n % 2 != 0:
            axarr[-1, -1].axis('off')
        f.set_size_inches(10, 30)

    f.suptitle(suptitle)    
    f.subplots_adjust(hspace=0.7)
    return f


def plot_parameters(X, delta, a):
    sns.set_context('paper')
    cmap1 = sns.cubehelix_palette(rot=-.5,light=1.5,dark=-.5,as_cmap=True)
    gs = gridspec.GridSpec(2, 2, height_ratios=[4, 2]) 
    f = plt.figure(figsize=(12,6))
    axarr = np.array([[None]*2]*2)
    for i in range(2):
        for j in range(2):
            axarr[i,j] = plt.subplot(gs[i*2+j])

    axarr[0, 0].scatter(X[:, 0], X[:, 1], c=delta, cmap=cmap1,
                        edgecolor='k',s=40)
    axarr[0, 0].set_title('$\mathbf{\delta}$ (Difficulty)',fontsize=16)

    
    axarr[0, 1].scatter(X[:, 0], X[:, 1], c=a, cmap=cmap1,
                        edgecolor='k',s=40)
    axarr[0, 1].set_title('$\mathbf{a}$ (Discrimination)',fontsize=16)

    #axarr[1, 0].hist(delta,bins=100)
    sns.distplot(delta,bins=100,ax=axarr[1,0])
    axarr[1, 0].set_title('Histogram of $\mathbf{\delta}$',fontsize=16)
    #axarr[1, 1].hist(a,bins=100)
    sns.distplot(a,bins=100,ax=axarr[1,1])
    axarr[1, 1].set_title('Histogram of $\mathbf{a}$',fontsize=16)
    f.suptitle('IRT item parameters')
    #f.set_size_inches(20, 20)
    f.subplots_adjust(hspace=0.3)
    return f

def plot_noisy_points(xtest, disc=None):
    sns.set_context('paper')
    cls = sns.color_palette("BuGn_r")
    lgd = []
    f = plt.figure()
    
    plt.scatter(xtest.x[xtest.noise==0],xtest.y[xtest.noise==0],facecolors='none',edgecolors='k',s=60)
    lgd.append('non-noise item')
    plt.scatter(xtest.x[xtest.noise>0],xtest.y[xtest.noise>0],c=cls[3],s=60)
    lgd.append('noise item')
    if not disc is None:
        plt.scatter(xtest.x[disc<0],xtest.y[disc<0],c=cls[0],marker='+',facecolors='none')
        lgd.append('detected noise item')
    
    plt.title('True and detected noise items')
    l = plt.legend(lgd,frameon=True,fontsize=12)
    l.get_frame().set_edgecolor('g')
    return f

def plot_item_parameters_corr(irt_prob_avg,difficulty,noise,disc=None):
    sns.set_context('paper')
    cls = sns.color_palette("BuGn_r")
    lgd = []

    f = plt.figure()
    plt.xlim([0.,1.])
    plt.ylim([0.,1.])
    
    
    plt.scatter(irt_prob_avg[noise>0],difficulty[noise>0],c=cls[3],s=60)
    lgd.append('noise item')
    if not disc is None:
        plt.scatter(irt_prob_avg[disc<0],difficulty[disc<0],c=cls[0],marker='+',facecolors='none')
        lgd.append('detected noise item')
    plt.scatter(irt_prob_avg[noise==0],difficulty[noise==0],facecolors='none',edgecolors='k',s=60)
    lgd.append('non-noise item')

    plt.title('Correlation between difficulty and response')
    plt.xlabel('Average response',fontsize=14)
    plt.ylabel('Difficulty',fontsize=14)
    l=plt.legend(lgd,frameon=True,fontsize=12)
    l.get_frame().set_edgecolor('g')
    return f


def vis_performance(gather_prec,gather_recal,path,asd='as1@5',vtype='nfrac'):
    fig = plt.figure()      
    plt.plot(gather_recal.index, gather_recal.mean(axis=1),marker='o')
    plt.plot(gather_prec.index, gather_prec.mean(axis=1),marker='^')

    plt.errorbar(gather_recal.index, gather_recal.mean(axis=1), gather_recal.std(axis=1), linestyle='None')
    plt.errorbar(gather_prec.index, gather_prec.mean(axis=1), gather_prec.std(axis=1), linestyle='None')
    
    if vtype=='nfrac':
        plt.title('Precision and recall under different noise fractions')
        plt.xlabel('Noise fraction (percentile)')
        plt.ylim(-0.05,1.1)
        plt.yticks(np.arange(0,1.2,0.2))
        plt.legend(['Recall','Precision'],loc=0)
        plt.savefig(path+'gathered_dnoise_performance_nfrac_'+asd+'.pdf') 
    elif vtype=='astd':
        plt.title('Precision and recall under different prior SD')
        plt.xlabel('Prior standard deviation of discrimination')
        plt.xlim(0.5,3.25)
        plt.ylim(-0.05,1.1)
        plt.yticks(np.arange(0,1.2,0.2))
        plt.legend(['Recall','Precision'],loc=0)
        plt.savefig(path+'gathered_dnoise_performance_asd_nfrac20.pdf')
    plt.close(fig)


def gather_vary_nfrac(path,dataset,a_prior_std=1.5,clcomb='79',mcomb='m10',idx = [2,5,10,20,30,40,50,55]):
    prefix = path+'dnoise_performance_'+dataset+'_s400_'
    files = glob.glob(prefix+'*.txt')
    #print(len(files))
    asd = 'as'+str(a_prior_std).replace('.','@')
    files = filter(lambda f:  '_'+mcomb+'_' in f and asd in f and 'cl'+clcomb in f , files)

    gather_prec = pd.DataFrame(index=idx)
    gather_recal = pd.DataFrame(index=idx)
    pfix1 = 'precision = '
    pfix2 = 'recall = '
    err_files = []
    for f in files:
        parse = re.split('_|\.',f[len(prefix)+1:])
        #print(parse)
        frac = int(parse[0])  
        #print(frac)
        if frac not in idx:
            continue
        seed = parse[1]

        with open(f,'r') as fr:
            l = fr.readlines()
            gather_prec.loc[frac,seed] = float(l[0][len(pfix1):])
            gather_recal.loc[frac,seed] = float(l[1][len(pfix2):])
        if np.isnan(gather_prec.loc[frac,seed]) or \
            np.isnan(gather_recal.loc[frac,seed]):
            print('find nan:',parse)
            err_files.append('./test_data/noise_test/'+dataset+'/bc4/'+mcomb+'/'+parse[2]+'/irt_data_'+dataset+'_s400_f'+parse[0]+'_'+parse[1]+'_'+parse[2]+'_'+mcomb+'.csv')
    return gather_prec,gather_recal,err_files

def vis_avg_all_clscombs_perform(dataset='mnist',a_prior_std=1.5,mcomb='m10',rpath='./results/bc4/mnist/m10/'):
    errs = []
    gather_precs=None
    gather_recals=None
    gather_prec_allcl = pd.DataFrame()
    gather_recal_allcl = pd.DataFrame()
    asd = 'as'+str(a_prior_std).replace('.','@')
    for i,cls in enumerate(combinations(np.arange(10),2)):
        #print(i)
        cl1, cl2 = cls[0],cls[1]
        comb = str(cl1)+str(cl2)
        path = rpath+'cl'+comb+'/'
        gather_prec,gather_recal, err = gather_vary_nfrac(path,dataset,a_prior_std,clcomb=comb,mcomb=mcomb)
        
        if len(err)==0:
            vis_performance(gather_prec,gather_recal,path,asd=asd)
        errs+=err
        if gather_precs is None:
            gather_precs = gather_prec
            gather_recals = gather_recal
            gather_prec_allcl = pd.DataFrame(index=gather_prec.index)
            gather_recal_allcl = pd.DataFrame(index=gather_recal.index)
        else:
            gather_precs+=gather_prec
            gather_recals+=gather_recal
        gather_prec_allcl[comb] = gather_prec.values.mean(axis=1)
        gather_recal_allcl[comb] = gather_recal.values.mean(axis=1)
    gather_precs /= i
    gather_recals /= i

    #vis_performance(gather_precs,gather_recals,rpath)
    vis_performance(gather_prec_allcl,gather_recal_allcl,rpath,asd=asd)
    if len(errs) > 0:
        with open('./retest.sh','w') as wf:
            for ef in errs:
                wf.writelines('python betairt_test.py '+ef+' a_prior_std:'+str(a_prior_std)+'\n')