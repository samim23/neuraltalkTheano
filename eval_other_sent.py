import argparse
import json
import time
import datetime
import numpy as np
import code
import os
import cPickle as pickle
import math
import scipy.io

from imagernn.solver import Solver
from imagernn.imagernn_utils import decodeGenerator, eval_split
from imagernn.data_provider import prepare_data, loadArbitraryFeatures
from picsom_bin_data import picsom_bin_data
from nltk.tokenize import word_tokenize

"""
This script is used to compute models opinion of how likely a given sentence corresponds to a given image
This can be used to build mutual evaluations of the different models with each other
"""

def main(params):

  # load the checkpoint
  checkpoint_path = params['checkpoint_path']
  print 'loading checkpoint %s' % (checkpoint_path, )
  checkpoint = pickle.load(open(checkpoint_path, 'rb'))
  checkpoint_params = checkpoint['params']

  model_npy = checkpoint['model']
  misc = {}
  misc['wordtoix'] = checkpoint['wordtoix']
  ixtoword = checkpoint['ixtoword']

  if 'use_theano' not in  checkpoint_params:
    checkpoint_params['use_theano'] = 1
  
  checkpoint_params['use_theano'] = 1

  if 'image_feat_size' not in  checkpoint_params:
    checkpoint_params['image_feat_size'] = 4096 

  # output blob which we will dump to JSON for visualizing the results
  blob = {} 
  blob['params'] = params
  blob['checkpoint_params'] = checkpoint_params
  blob['imgblobs'] = []

  # load the tasks.txt file
  root_path = params['root_path']
  img_names_list = open(params['imgList'], 'r').read().splitlines()

  if len(img_names_list[0].rsplit(',')) > 2: 
    img_names = [x.rsplit (',')[0] for x in img_names_list]
    sentRaw = [x.rsplit (',')[1] for x in img_names_list]
    idxes = [int(x.rsplit (',')[2]) for x in img_names_list]
  elif len(img_names_list[0].rsplit(',')) == 2:
    img_names = [x.rsplit (',')[0] for x in img_names_list]
    sentRaw = [x.rsplit (',')[1] for x in img_names_list]
    idxes = xrange(len(img_names_list))
  else:
    print 'ERROR: List should atleast contain image name and a corresponding sentence'
    return

  if checkpoint_params.get('en_aux_inp',0) and (params.get('aux_inp_file',None) == None):
    raise ValueError('ERROR: please specify auxillary input feature using --aux_inp_file')
    return
  # load the features for all images
  features, aux_inp = loadArbitraryFeatures(params, idxes)

  D,NN = features.shape
  N = len(img_names) 

  # iterate over all images and predict sentences
  BatchGenerator = decodeGenerator(checkpoint_params)
  BatchGenerator.build_eval_other_sent(BatchGenerator.model_th, checkpoint_params,model_npy)
  eval_batch_size = params.get('eval_batch_size',100)
  wordtoix = checkpoint['wordtoix']
  
  gen_fprop = BatchGenerator.f_eval_other
  
  print("\nUsing model run for %0.2f epochs with validation perplx at %0.3f\n" % (checkpoint['epoch'], \
    checkpoint['perplexity']))
  
  n = 0
  
  while n < N:
    print('image %d/%d:\r' % (n, N)),
    
    cbs = 0
    # encode the image
    batch = []
    while n < N and cbs < eval_batch_size:
        out = {}
        out['image'] = {'feat':features[:, n]}
        out['sentence'] = {'raw': sentRaw[n],'tokens':word_tokenize(sentRaw[n])}
        out['idx'] = n
        if checkpoint_params.get('en_aux_inp',0):
            out['image']['aux_inp'] = aux_inp[:, n]

        cbs += 1
        n += 1
        batch.append(out)
    
    inp_list, lenS = prepare_data(batch,wordtoix)

    # perform the work. heavy lifting happens inside
    eval_array = gen_fprop(*inp_list)

    for ix,x in enumerate(batch):
        # build up the output
        img_blob = {}
        img_blob['img_path'] = img_names[x['idx']]
        # encode the top prediction
        img_blob['candidate'] = {'text': x['sentence']['raw'], 'logprob': float(eval_array[0,ix])}
        blob['imgblobs'].append(img_blob)

  # dump result struct to file
  jsonFname = 'result_struct_%s.json' % (params['fname_append'] ) 
  save_file = os.path.join(root_path, jsonFname)
  print 'writing predictions to %s...' % (save_file, )
  json.dump(blob, open(save_file, 'w'))

  # dump output html

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('checkpoint_path', type=str, help='the input checkpoint')
  parser.add_argument('-i', '--imgList', type=str, default='testimgs.txt', help='file with the list of images to process. Either just filenames or in <filenmae, index> format')
  parser.add_argument('-f', '--feat_file', type=str, default='vgg_feats.mat', help='file with the features. We can rightnow process only .mat format') 
  parser.add_argument('-d', '--dest', dest='root_path', default='example_images', type=str, help='folder to store the output files')
  parser.add_argument('-b', '--beam_size', type=int, default=1, help='beam size in inference. 1 indicates greedy per-word max procedure. Good value is approx 20 or so, and more = better.')
  parser.add_argument('--fname_append', type=str, default='', help='str to append to routput files')
  parser.add_argument('--aux_inp_file', dest='aux_inp_file', type=str, default=None, help='Is there any auxillary inputs ? If yes indicate file here')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  
  if params['aux_inp_file'] != None:
    params['en_aux_inp'] = 1
  else:
    params['en_aux_inp'] = 0

  print 'parsed parameters:'
  print json.dumps(params, indent = 2)
  main(params)
