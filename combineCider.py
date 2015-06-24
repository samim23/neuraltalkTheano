import argparse
import json
import os
import random
import scipy.io
import codecs
import numpy as np
import cPickle as pickle
from collections import Counter
from collections import defaultdict 
from nltk.tokenize import word_tokenize
from eval.mseval.pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
import re
import math
from time import time

def precook(s, n=4):
    """
    Takes a string as input and returns an object that can be given to
    either cook_refs or cook_test. This is optional: cook_refs and cook_test
    can take string arguments as well.
    :param s: string : sentence to be converted into ngrams
    :param n: int    : number of ngrams for which representation is calculated
    :return: term frequency vector for occuring ngrams
    """
    words = s.split()
    counts = Counter()
    for k in xrange(1,n+1):
        for i in xrange(len(words)-k+1):
            ngram = '_'.join(words[i:i+k])
            counts[ngram] += 1
    return counts

def counts2vec(cnts, n, dfGlb, Nrefs):
    """
    Function maps counts of ngram to vector of tfidf weights.
    The function returns vec, an array of dictionary that store mapping of n-gram and tf-idf weights.
    The n-th entry of array denotes length of n-grams.
    :param cnts:
    :return: vec (array of dict), norm (array of float), length (int)
    """
    vec = [defaultdict(float) for _ in range(n)]
    length = 0
    norm = [0.0 for _ in range(n)]
    for (ngram,term_freq) in cnts.iteritems():
        # give word count 1 if it doesn't appear in reference corpus
        df = np.log(max(1.0, dfGlb[ngram]))
        # ngram index
        n = len(ngram.split('_'))-1
        # tf (term_freq) * idf (precomputed idf) for n-grams
        vec[n][ngram] = float(term_freq)*(np.log(float(Nrefs)) - df)
        # compute norm for the vector.  the norm will be used for computing similarity
        norm[n] += pow(vec[n][ngram], 2)

        if n == 0:
            length += term_freq
    norm = [np.sqrt(n) for n in norm]
    return vec, norm, length
        
def sim(vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref, ngMax=4, sigma=6):
    '''
    Compute the cosine similarity of two vectors.
    :param vec_hyp: array of dictionary for vector corresponding to hypothesis
    :param vec_ref: array of dictionary for vector corresponding to reference
    :param norm_hyp: array of float for vector corresponding to hypothesis
    :param norm_ref: array of float for vector corresponding to reference
    :param length_hyp: int containing length of hypothesis
    :param length_ref: int containing length of reference
    :return: array of score for each n-grams cosine similarity
    '''
    delta = float(length_hyp - length_ref)
    # measure consine similarity
    val = np.array([0.0 for _ in range(ngMax)])
    
    for n in xrange(ngMax):
        for ngram in vec_hyp[n]:
            # vrama91 : added clipping
            if ngram in vec_ref[n]:
                val[n] += vec_hyp[n][ngram] * vec_ref[n][ngram]

        if (norm_hyp[n] != 0) and (norm_ref[n] != 0):
            val[n] /= (norm_hyp[n]*norm_ref[n])

        # vrama91: added a length based gaussian penalty
        val[n] *= np.e**(-(delta**2)/(2*sigma**2))
        
    return np.mean(val)


def getTfIdfWeights(params):

  if params.get('tfIdf_file','') == '':
     dataset = json.load(open('/triton/ics/project/imagedb/picsom/databases/COCO/download/annotations/captions_train2014.json','r'))
     tokenizer = PTBTokenizer()
     origRefs = {}
     curr_keys = set()
     n = params.get('max_ngram',4)
     for anns in dataset['annotations']:
         if anns['image_id'] not in curr_keys:
             origRefs[anns['image_id']] = []
             curr_keys.add(anns['image_id'])
         origRefs[anns['image_id']].append(anns)
      
     origRefs  = tokenizer.tokenize(origRefs)

     doc_freq = Counter()
     for refs in origRefs.iteritems():
         rCounter = Counter()
         for s in refs[1]:
             rCounter += precook(s,n)
         for ngrams in rCounter.keys():
             doc_freq[ngrams] += 1 
     
     tfidf = {'doc_freq':doc_freq,'N':len(origRefs)}
     pickle.dump(tfidf,open('tf_idf_ngrams_4_allTrnRefs.p','w'))
  else:
      tfidf = pickle.load(open(params.get('tfIdf_file'),'r'))

  return tfidf 

def computeCiderScrs(cands,tfidf,n=4):

  ncands = len(cands)
  hKCnts = []
  vec = []
  norm = []
  length = []
  
  for i,s in enumerate(cands):
     hKCnts.append(precook(s,n))
     vecL,normL,lengthL = counts2vec(hKCnts[i],n,tfidf['doc_freq'],tfidf['N'])
     vec.append(vecL)
     norm.append(normL)
     length.append(lengthL)
  
  score = np.zeros((ncands,ncands))
  
  for i in xrange(len(hKCnts)):
    for j in xrange(i+1,len(hKCnts)):
        score[i,j]= sim(vec[i], vec[j], norm[i], norm[j], length[i], length[j], n)
        score[j,i] = score[i,j]
  
  cidScrs = score.sum(axis=0)
  
  return cidScrs



def pickOnMutualCideR(params,tfidf):
  
  candDataset = pickle.load(open(params['cand_dB'],'r'))
  
  n = params.get('max_ngram',4)
  bestCandRes = []
  for imgId, img in enumerate(candDataset['imgblobs']):
      #import pdb;pdb.set_trace()
      cidScrs = computeCiderScrs(img['cands'],tfidf,n)

      candDataset['imgblobs'][imgId]['mciderAll'] = cidScrs
      bestIdx = np.argmax(cidScrs)
      bestCandRes.append({'image_id':int(img['imgid']), 'caption':img['cands'][bestIdx].lstrip(' ').rstrip(' ')})
      if imgId%500 == 1:
        print('i = %d'%imgId)
   
  json.dump(bestCandRes,open(params['outfile'],'w'))
  pickle.dump(candDataset,open(params['cand_dB'],'w'))

def pickOnMutualCideRTopk(params,tfidf,k=10,prevScrs='mciderAll'):
  
  candDataset = pickle.load(open(params['cand_dB'],'r'))
  
  newScrs = 'mciderTop' + str(k)
  newScrsIdx = 'srtIdxTop' + str(k)
  n = params.get('max_ngram',4)
  bestCandRes = []
  
  for imgId, img in enumerate(candDataset['imgblobs']):
      #import pdb;pdb.set_trace()
      sortIdx = np.argsort(img[prevScrs])[::-1][:k]
      cands = [img['cands'][si] for si in sortIdx]
      
      cidScrs = computeCiderScrs(cands,tfidf,n)

      candDataset['imgblobs'][imgId][newScrs] = cidScrs
      candDataset['imgblobs'][imgId][newScrsIdx] = sortIdx 
      bestIdx = np.argmax(cidScrs)
      bestCandRes.append({'image_id':int(img['imgid']), 'caption':cands[bestIdx].lstrip(' ').rstrip(' ')})
      if imgId%500 == 1:
        print('i = %d'%imgId)
   
  json.dump(bestCandRes,open(params['outfile'],'w'))
  pickle.dump(candDataset,open(params['cand_dB'],'w'))
        

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('--resdir', type=str,  default='', help='The root directory from which to gather candidate captions from')
  parser.add_argument('--cand_dB', type=str, default='', help='filename of the result struct to save')
  parser.add_argument('--tfIdf_file', type=str, default='', help='filename to read TF IDF weights for ngrams from')
  parser.add_argument('--outfile', type=str, default='captions_Cider.json', help='filename to read TF IDF weights for ngrams from')
  
  parser.add_argument('--repeatTopk', type=int, default=0, help='filename to read TF IDF weights for ngrams from')
  parser.add_argument('--prevScr', type=str, default='mciderAll', help='filename to read TF IDF weights for ngrams from')


  
  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  
  tfidf = getTfIdfWeights(params)
  if params['repeatTopk'] == 0:
    pickOnMutualCideR(params,tfidf)
  else:
    pickOnMutualCideRTopk(params,tfidf,params['repeatTopk'],params['prevScr'])
  #evaluate_decision(params, com_dataset, eval_array)
