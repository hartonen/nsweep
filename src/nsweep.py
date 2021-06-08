#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import pyfastx
import csv
import argparse
import deeplift

from deeplift.conversion import kerasapi_conversion as kc

import numpy as np
import pandas as pd
import multiprocessing as mp
import logomaker as lm

from time import time

def nsweep():

    ########################
    #command line arguments#
    ########################

    parser = argparse.ArgumentParser()

    #PARAMETERS
    parser.add_argument("--pfms",help="Full path(s) to the pfm-files containing the tested motifs.",type=str,nargs='+')
    parser.add_argument("--pfmnames",help="Names of the PFMs, must be in same order as the input files.",type=str,nargs='+')
    parser.add_argument("--outdir",help="Full path to the output directory.",type=str)
    parser.add_argument("--references",help="Full path to a fasta-file containing the sequences used as reference when calculating Deeplift contribution scores. If the fasta-file contains more than one sequence, contributions are averaged over different backgrounds.",type=str)
    parser.add_argument("--model",help="Full path to the trained keras model (.h5 format).",type=str,default=None)
    parser.add_argument("--background",help="Full path to a fasta-file containing the background sequences where each PFM is embedded. If there are more than one background sequence, contributions are calculated as an average over the set of sequences where each motif is embedded to each background sequence (in each position defined by flag --positions).",type=str)
    parser.add_argument("--positions",help="If random (=default), each pfm is embedded to --N random positions per background sequence. If you want to specify the positions for embedding, use this flag to give input file that contains all positions for embedding (each position on its own line, 0-based coordinates).",type=str,default="random")
    parser.add_argument("--N",help="Number of random positions drawn per background sequence for embedding a pfm (default=100).",type=int,default=100)
    parser.add_argument("--target_layer",help="Target layer index for deeplift (default=-3).",type=int,default=-3)
    parser.add_argument("--ylim",help="Limits for y-axis.",type=float,nargs=2,default=None)
    parser.add_argument("--logoType",help="Logo image file extension (default=pdf).",type=str,default='pdf',choices=['png','pdf'])

    args = parser.parse_args()

    
    #first read in the reference and the background sequences
    references_onehot = []
    background = []
    for seq in pyfastx.Fasta(args.references): references_onehot.append(vectorizeSequence(str(seq.seq).upper()))
    for seq in pyfastx.Fasta(args.background): background.append(str(seq.seq).upper())
    N_ref = len(references_onehot)
    
    #one-hot encode the reference sequences
    

    
    #then read in the Keras model and convert to Deeplift model
    #initialize the deeplift model
    deeplift_model = kc.convert_model_from_saved_files(args.model,nonlinear_mxts_mode=deeplift.layers.NonlinearMxtsMode.DeepLIFT_GenomicsDefault)
    find_scores_layer_idx = 0 #computes importance scores for inpur layer input
    deeplift_contribs_func = deeplift_model.get_target_contribs_func(find_scores_layer_idx=find_scores_layer_idx,target_layer_idx=args.target_layer)

    #then analyze each PFM sequentially
    pfmind = -1
    for pfmfile in args.pfms:
        start = time
        pfmind += 1
        pfm = np.loadtxt(pfmfile)
        cwm = np.zeros(shape=pfm.shape)
        #define consensus sequence
        consensus_inds = np.argmax(pfm,axis=0)
        consensus_seq = ""
        for i in consensus_inds:
            if i==0: consensus_seq += 'A'
            elif i==1: consensus_seq += 'C'
            elif i==2: consensus_seq += 'G'
            elif i==3: consensus_seq += 'T'
        #draw start positions for embedding or read from file
        if args.positions=='random': positions = np.random.randint(0,high=len(background[0])-len(consensus_seq),size=args.N)
        else: positions = np.loadtxt(args.positions)
        

        #compute contribution scores for each column of the PFM
        for i in range(pfm.shape[0]):
            variants = [consensus_seq[:i]+l+consensus_seq[i+1:] for l in ['A','C','G','T']]
            for variant in variants:
                letter = variant[i]
                #embed variant to all background+position combinations
                seqs = []
                seqs_onehot = []
                for seq in background:
                    for p in positions:
                        seqs.append(seq[:p]+variant+seq[p+len(variant):])
                        seqs_onehot.append(vectorizeSequence(seqs[-1]))
                seqs_onehot = np.array(seqs)    
                #and then score each sequence against all different reference sequences
                scores = np.zeros(shape=(N_ref,seqs_onehot.shape[0],seqs_onehot.shape[1]))
                for n in range(N_ref):
                    scores[n,:,:] = np.sum(deeplift_contribs_func(task_idx=1,input_data_list=[seqs_onehot],input_references_list=[references_onehot[:seqs_onehot.shape[0],:,:]],batch_size=10,progress_update=None),axis=2)
                    references_onehot = np.roll(bg,1,axis=0)
                                           
                scores = np.mean(scores,axis=0) #compute average over the different reference sequences
                #set the CWM values for column i
                count = 0.0
                for l in range(seqs_onehot.shape[0]):
                    for m in range(len(positions)):
                        count += 1
                        pos = positions[l]+i
                        
                        if letter=='A': cwm[i,0] += scores[pos,0]
                        elif letter=='C': cwm[i,1] += scores[pos,1]
                        elif letter=='G': cwm[i,2] += scores[pos,2]
                        elif letter=='T': cwm[i,3] += scores[pos,3]
                #use the average contribution as final measure
                cwm[i,:] /= count
        #Now the CWM is computed, saving and plotting the results                                   
        np.savetxt(args.outdir+'-'+args.pfmnames[pfmind]+'.cwm',cwm)
        plotLogo(cwm,'position','contribution',args.outdir+'-'+args.pfmnames[pfmind]+'.png')
        end = time()
        print("Processed "+args.pfmnames[pfmind]+" in "+str(end-start)+" s")
                    
#end

def vectorizeSequence(seq):
    # the order of the letters is not arbitrary.
    # Flip the matrix up-down and left-right for reverse complement
    ltrdict = {'A':[1,0,0,0],'C':[0,1,0,0],'G':[0,0,1,0],'T':[0,0,0,1]}
    return np.array([ltrdict[x] for x in seq])

def plotLogo(matrix,xlabel,ylabel,outfile):
    #plots a logo of matrix using logoMaker

    fig,ax = plt.subplots()
    
    matrix_df = pd.DataFrame(matrix.transpose())
    matrix_df.columns = ['A','C','G','T']
    logo = lm.Logo(df=matrix_df,color_scheme='classic')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.savefig(outfile,dpi=150)
    plt.close(fig)
    plt.clf()
    plt.cla()
    #print("done!")
    return True

nsweep()
