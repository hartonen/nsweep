#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import pyfastx
import csv
import argparse
import deeplift

from deeplift.conversion import kerasapi_conversion as kc
from deeplift.util import get_shuffle_seq_ref_function
from deeplift.dinuc_shuffle import dinuc_shuffle #function to do a dinucleotide shuffle


import numpy as np
import pandas as pd
import multiprocessing as mp
import logomaker as lm

from time import time
from Bio import SeqIO

import keras
from keras.models import load_model

def nsweep():

    ########################
    #command line arguments#
    ########################

    parser = argparse.ArgumentParser()

    #PARAMETERS
    parser.add_argument("--pfms",help="Full path(s) to the pfm-files containing the tested motifs.",type=str,nargs='+')
    parser.add_argument("--pfmnames",help="Names of the PFMs, must be in same order as the input files.",type=str,nargs='+')
    parser.add_argument("--outdir",help="Full path to the output directory.",type=str)
    parser.add_argument("--references",help="Full path to a fasta-file containing the sequences used as reference when calculating Deeplift contribution scores. Note that due to implementation reasons, the number of reference sequences used needs to be higher than number of background sequences times number of embedding positions.",type=str)
    parser.add_argument("--model",help="Full path to the trained keras model (.h5 format).",type=str,default=None)
    parser.add_argument("--background",help="Full path to a fasta-file containing the background sequences where each PFM is embedded. If there are more than one background sequence, contributions are calculated as an average over the set of sequences where each motif is embedded to each background sequence (in each position defined by flag --positions).",type=str)
    parser.add_argument("--positions",help="If random (=default), each pfm is embedded to --N random positions per background sequence. If you want to specify the positions for embedding, use this flag to give input file that contains all positions for embedding (each position on its own line, 0-based coordinates).",type=str,default="random")
    parser.add_argument("--N",help="Number of random positions drawn per background sequence for embedding a pfm (default=100).",type=int,default=100)
    parser.add_argument("--Nref",help="Number of references used per sequence when computing Deeplift importances (default=10).",type=int,default=10)
    parser.add_argument("--target_layer",help="Target layer index for deeplift (default=-3).",type=int,default=-3)
    parser.add_argument("--ylim",help="Limits for y-axis.",type=float,nargs=2,default=None)
    parser.add_argument("--logoType",help="Logo image file extension (default=pdf).",type=str,default='pdf',choices=['png','pdf'])

    args = parser.parse_args()

    
    #first read in the reference and the background sequences
    references_onehot = []
    #references = []
    for fasta in SeqIO.parse(open(args.references),'fasta'): references_onehot.append(vectorizeSequence(str(fasta.seq).upper()))
    
    background = []
    for fasta in SeqIO.parse(open(args.background),'fasta'): background.append(str(fasta.seq).upper())
    #for seq in pyfastx.Fasta(args.references):
    #    print(seq)
    #    print(seq.seq)
    #    references.append(str(seq.seq).upper())
    #    #references_onehot.append(vectorizeSequence(str(seq.seq).upper()))
    #for seq in pyfastx.Fasta(args.background): background.append(str(seq.seq).upper())
    #N_ref = len(references_onehot)
    #for seq in references: references_onehot.append(vectorizeSequence(seq))
    references_onehot = np.array(references_onehot)
    print("references_onehot.shape="+str(references_onehot.shape))
    #while True:
    #    z = input('any')
    #    break
    
    #one-hot encode the reference sequences
    

    
    #then read in the Keras model and convert to Deeplift model
    #initialize the deeplift model
    keras_model = load_model(args.model)
    deeplift_model = kc.convert_model_from_saved_files(args.model,nonlinear_mxts_mode=deeplift.layers.NonlinearMxtsMode.DeepLIFT_GenomicsDefault)
    find_scores_layer_idx = 0 #computes importance scores for inpur layer input
    deeplift_contribs_func = deeplift_model.get_target_contribs_func(find_scores_layer_idx=find_scores_layer_idx,target_layer_idx=args.target_layer)

    contribs_many_refs_func = get_shuffle_seq_ref_function(
    #score_computation_function is the original function to compute scores
    score_computation_function=deeplift_contribs_func,
    #shuffle_func is the function that shuffles the sequence
    #On real genomic data, a dinuc shuffle is advisable due to
    #the strong bias against CG dinucleotides
    shuffle_func=dinuc_shuffle,
    one_hot_func=lambda x: np.array([one_hot_encode_along_channel_axis(seq) for seq in x]))
    
    #then analyze each PFM sequentially
    pfmind = -1
    for pfmfile in args.pfms:
        start = time()
        pfmind += 1
        pfm = np.loadtxt(pfmfile)
        cwm = np.zeros(shape=(pfm.shape[1],pfm.shape[0])) #contribution weight matrix from Deeplift contributions
        pwm = np.zeros(shape=cwm.shape) #probability weight matrix from predicted enhancer probabilities
        alignment = np.zeros(shape=cwm.shape) #simple alignment of Deeplift scores for single nucleotide variants of consensus
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
        else:
            positions = []#list(np.loadtxt(args.positions,dtype=int))
            with open(args.positions,'rt') as infile:
                r = csv.reader(infile,delimiter='\t')
                for row in r: positions.append(int(row[0]))
        

        #compute contribution scores for each column of the PFM
        for i in range(pfm.shape[1]):
            variants = [consensus_seq[:i]+l+consensus_seq[i+1:] for l in ['A','C','G','T']]
            print("pos="+str(i))
            seqs = []
            seqs_onehot = []
            embed_positions = []
            background_seqs = []
            background_seqs_onehot = []
            for variant in variants:
                letter = variant[i]
                print("variant="+variant+", letter="+letter)
                #embed variant to all background+position combinations
                for seq in background:
                    for p in positions:
                        seqs.append(seq[:p]+variant+seq[p+len(variant):])
                        seqs_onehot.append(vectorizeSequence(seqs[-1]))
                        embed_positions.append(p)
                        background_seqs.append(seq)
                        background_seqs_onehot.append(vectorizeSequence(seq))
            seqs_onehot = np.array(seqs_onehot)
            background_seqs_onehot = np.array(background_seqs_onehot)
            #and then score each sequence against all different reference sequences
            #print(seqs_onehot.shape)
            #print(seqs_onehot[0])
            deeplift_scores = np.zeros(shape=(seqs_onehot.shape[0],seqs_onehot.shape[1]))
            #dimensions of scores
            #0 = number of scored sequences
            #1 = length of sequences
            print("deeplift_scores.shape="+str(deeplift_scores.shape))

            deeplift_scores = np.sum(contribs_many_refs_func(task_idx=1,input_data_sequences=seqs,num_refs_per_seq=args.Nref,batch_size=50,progress_update=None),axis=2)#[:,:,None]*seqs_onehot
            print("deeplift_scores.shape="+str(deeplift_scores.shape))
            keras_scores = keras_model.predict(np.stack(seqs_onehot))[:,1]
            keras_background_scores = keras_model.predict(np.stack(background_seqs_onehot))[:,1]
            keras_scores = (keras_scores-keras_background_scores)/keras_background_scores

            #construct the matrix columns
            totcount = 0.0
            for j in range(len(seqs)):
                mutant_pos = embed_positions[j]+i
                cwm[i,:] += deeplift_scores[j,mutant_pos]*seqs_onehot[j,mutant_pos,:]
                pwm[i,:] += keras_scores[j]*seqs_onehot[j,mutant_pos,:]
                #alignment[:,:] += deeplift_scores[j,embed_positions[j]:embed_positions[j]+pfm.shape[1]]*seqs_onehot[j,embed_positions[j]:embed_positions[j]+pfm.shape[1],:]
                totcount += 1.0
            cwm[i,:] /= (len(seqs)/4.0)#np.sum(np.abs(cwm[i,:]))#(len(seqs)/4.0)
            pwm[i,:] /= np.sum(np.abs(pwm[i,:]))#(len(seqs)/4.0)
        #entire matrices computed, normalize and save
        #alignment /= totcount
        
        np.savetxt(args.outdir+args.pfmnames[pfmind]+'.cwm',cwm)
        plotLogo(cwm,'position','contribution',args.outdir+args.pfmnames[pfmind]+'-cwm.'+args.logoType)
        np.savetxt(args.outdir+args.pfmnames[pfmind]+'.pwm',pwm)
        plotLogo(pwm,'position','P_diff',args.outdir+args.pfmnames[pfmind]+'-pwm.'+args.logoType)
        #np.savetxt(args.outdir+args.pfmnames[pfmind]+'.alignment',alignment)
        #plotLogo(alignment,'position','contribution',args.outdir+args.pfmnames[pfmind]+'-alignment.'+args.logoType)
        end = time()
        print("Processed "+args.pfmnames[pfmind]+" in "+str(end-start)+" s")
        """
        
            
            #for n in range(args.Nref):
            #    scores[n,:,:] = np.sum(deeplift_contribs_func(task_idx=1,input_data_list=[seqs_onehot],input_references_list=[references_onehot[:seqs_onehot.shape[0],:,:]],batch_size=10,progress_update=None),axis=2)
            #    references_onehot = np.roll(references_onehot,1,axis=0)

                #as an additional test, score each sequence using the keras model and show predicted probabilities
                P = keras_model.predict(np.stack(seqs_onehot))[:,1]
                print("Scores before taking mean for column "+str(i))
                print(scores[:,:,0:15])
                print("Model-predicted probabilities before mean:")
                print(P)
                scores = np.mean(scores,axis=0) #compute average over the different reference sequences¨

                #now the contributions have been calculated, next plotting the sequence logos weighted by the contributions for debugging purposes
                #ind = 0
                #for seq in background:
                #    for p in positions:
                #    #first plotting the sequence
                #        seq = seqs[ind]
                #        fig, ax = plt.subplots()
                #        matrix_df = lm.saliency_to_matrix(seq,scores[ind,:])#pd.DataFrame(scores[i,:])
                #        logo = lm.Logo(df=matrix_df,color_scheme='classic')
                #        logo.ax.set_xlabel('position')
                #        logo.ax.set_ylabel('contribution')
                #        title = "bg_ind="+str(ind)+"-variant="+variant+"-letter="+letter+"-position="+str(p)
                #        logo.ax.set_title(title)
                #        if args.ylim!=None: logo.ax.set_ylim(args.ylim)
                #        plt.tight_layout()
                #        plt.savefig(args.outdir+title+"."+args.logoType,dpi=150,bbox_inches='tight',pad_inches=0) 
                #        plt.close(fig)
                #        plt.clf()
                #        plt.cla()
                #        ind += 1
                #dimensions of scores
                #0 = number of background sequences
                #1 = length of sequences
                #2 = number of possible nucleotides
                
                #set the CWM values for column i
                #if args.positions=="random": count = len(background)*args.N
                #else: count = len(background)*positions.shape[0] 
                print("seqs_onehot.shape="+str(seqs_onehot.shape))
                print("pfm.shape="+str(pfm.shape))
                print("cwm.shape="+str(cwm.shape))
                print("scores.shape="+str(scores.shape))
                print(scores)
                #while True:
                #    z = input("any")
                #    break
                for l in range(seqs_onehot.shape[0]):
                    for m in range(len(positions)):
                        pos = positions[m]+i #This is the mutated position corresponding to i in consensus sequence of pfm
                        print("contributions="+str(np.transpose(scores[l,pos])))
                        print("seq_onehot="+str(seqs_onehot[l,pos,:]))
                        print("seq="+str(seqs[l][pos]))
                        cwm[:,i] += np.transpose(scores[l,pos]*seqs_onehot[l,pos,:])
                        pwm[:,i] += np.transpose(P[l]*seqs_onehot[l,pos,:])
                #while True:
                #    z = input("any")
                #    break
                #for l in range(seqs_onehot.shape[0]):
                #    for m in range(len(positions)):
                #        #count += 1
                #        pos = positions[m]+i
                #        
                #        if letter=='A': cwm[0,i] += scores[l,pos,0]
                #        elif letter=='C': cwm[1,i] += scores[l,pos,1]
                #        elif letter=='G': cwm[2,i] += scores[l,pos,2]
                #        elif letter=='T': cwm[3,i] += scores[l,pos,3]
                #use the average contribution as final measure
        if args.positions=="random": count = len(background)*args.N
        else: count = len(background)*len(positions)        
        cwm[:,i] /= count
        pwm[:,i] /= count
        #Now the CWM is computed, saving and plotting the results                                   
        np.savetxt(args.outdir+args.pfmnames[pfmind]+'.cwm',cwm)
        plotLogo(cwm,'position','contribution',args.outdir+args.pfmnames[pfmind]+'-cwm.'+args.logoType)
        np.savetxt(args.outdir+args.pfmnames[pfmind]+'.pwm',pwm)
        plotLogo(pwm,'position','contribution',args.outdir+args.pfmnames[pfmind]+'-pwm.'+args.logoType)
        end = time()
        print("Processed "+args.pfmnames[pfmind]+" in "+str(end-start)+" s")
        """                
#end

def vectorizeSequence(seq):
    # the order of the letters is not arbitrary.
    # Flip the matrix up-down and left-right for reverse complement
    ltrdict = {'A':[1,0,0,0],'C':[0,1,0,0],'G':[0,0,1,0],'T':[0,0,0,1]}
    return np.array([ltrdict[x] for x in seq])

def one_hot_encode_along_channel_axis(sequence):
    to_return = np.zeros((len(sequence),4), dtype=np.int8)
    seq_to_one_hot_fill_in_array(zeros_array=to_return,
                                 sequence=sequence, one_hot_axis=1)
    return to_return

def seq_to_one_hot_fill_in_array(zeros_array, sequence, one_hot_axis):
    assert one_hot_axis==0 or one_hot_axis==1
    if (one_hot_axis==0):
        assert zeros_array.shape[1] == len(sequence)
    elif (one_hot_axis==1): 
        assert zeros_array.shape[0] == len(sequence)
    #will mutate zeros_array
    for (i,char) in enumerate(sequence):
        if (char=="A" or char=="a"):
            char_idx = 0
        elif (char=="C" or char=="c"):
            char_idx = 1
        elif (char=="G" or char=="g"):
            char_idx = 2
        elif (char=="T" or char=="t"):
            char_idx = 3
        elif (char=="N" or char=="n"):
            continue #leave that pos as all 0's
        else:
            raise RuntimeError("Unsupported character: "+str(char))
        if (one_hot_axis==0):
            zeros_array[char_idx,i] = 1
        elif (one_hot_axis==1):
            zeros_array[i,char_idx] = 1
            
def plotLogo(matrix,xlabel,ylabel,outfile):
    #plots a logo of matrix using logoMaker

    fig,ax = plt.subplots()
    
    matrix_df = pd.DataFrame(matrix)#.transpose())
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
