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
from picsom_bin_data import picsom_bin_data

"""
This script is used to predict sentences for arbitrary images
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

  if len(img_names_list[0].rsplit(',')) > 1: 
    img_names = [x.rsplit (',')[0] for x in img_names_list]
    idxes = [int(x.rsplit (',')[1]) for x in img_names_list]
  else:
    img_names = img_names_list
    idxes = xrange(len(img_names_list))

  # load the features for all images
  features_path = params['feat_file']
  if features_path.rsplit('.',1)[1] == 'mat':
    features_struct = scipy.io.loadmat(features_path)
    features = features_struct['feats'][:,idxes] # this is a 4096 x N numpy array of features
  else:
    features_struct = picsom_bin_data(features_path) 
    features = np.array(features_struct.get_float_list(idxes)).T; # this is a 4096 x N numpy array of features
    print "Working on Bin file now"

  D,NN = features.shape
  N = len(img_names) 

  # iterate over all images and predict sentences
  BatchGenerator = decodeGenerator(checkpoint_params)
  
  if checkpoint_params['use_theano'] == 1:
  	# Compile and init the theano predictor 
    BatchGenerator.prepPredictor(model_npy, checkpoint_params, params['beam_size'])
    model = BatchGenerator.model_th
    print("\nUsing model run for %0.2f epochs with validation perplx at %0.3f\n" % (checkpoint['epoch'], \
      checkpoint['perplexity']))
  
  kwparams = { 'beam_size' : params['beam_size'] }
  
  for n in xrange(N):
    print 'image %d/%d:' % (n, N)

    # encode the image
    img = {}
    img['feat'] = features[:, n]
    img['local_file_path'] =img_names[n]

    # perform the work. heavy lifting happens inside
    Ys = BatchGenerator.predict([{'image':img}], model, checkpoint_params, **kwparams)
    print Ys

    # build up the output
    img_blob = {}
    img_blob['img_path'] = img['local_file_path']

    # encode the top prediction
    top_predictions = Ys[0] # take predictions for the first (and only) image we passed in
    top_prediction = top_predictions[0] # these are sorted with highest on top
    candidate = ' '.join([ixtoword[int(ix)] for ix in top_prediction[1] if ix > 0]) # ix 0 is the END token, skip that
    print 'PRED: (%f) %s' % (float(top_prediction[0]), candidate)
    img_blob['candidate'] = {'text': candidate, 'logprob': float(top_prediction[0])}    

    # Code to save all the other candidates 
    candlist = []
    for ci in xrange(len(top_predictions)-1):
        prediction = top_predictions[ci+1] # these are sorted with highest on top
        candidate = ' '.join([ixtoword[int(ix)] for ix in prediction[1] if ix > 0]) # ix 0 is the END token, skip that
        candlist.append({'text': candidate, 'logprob': float(prediction[0])})
    
    img_blob['candidatelist'] = candlist


    blob['imgblobs'].append(img_blob)

  # dump result struct to file
  jsonFname = 'result_struct_%s.json' % (params['fname_append'] ) 
  save_file = os.path.join(root_path, jsonFname)
  print 'writing predictions to %s...' % (save_file, )
  json.dump(blob, open(save_file, 'w'))

  # dump output html
  html = ''
  for img in blob['imgblobs']:
    html += '<img src="%s" height="400"><br>' % (img['img_path'], )
    html += '(%f) %s <br><br>' % (img['candidate']['logprob'], img['candidate']['text'])

  html_file = 'result_%s.html' % (params['fname_append']) 
  html_file = os.path.join(root_path, html_file)
  print 'writing html result file to %s...' % (html_file, )
  open(html_file, 'w').write(html)

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('checkpoint_path', type=str, help='the input checkpoint')
  parser.add_argument('-i', '--imgList', type=str, default='testimgs.txt', help='file with the list of images to process. Either just filenames or in <filenmae, index> format')
  parser.add_argument('-f', '--feat_file', type=str, default='vgg_feats.mat', help='file with the features. We can rightnow process only .mat format') 
  parser.add_argument('-d', '--dest', dest='root_path', default='example_images', type=str, help='folder to store the output files')
  parser.add_argument('-b', '--beam_size', type=int, default=1, help='beam size in inference. 1 indicates greedy per-word max procedure. Good value is approx 20 or so, and more = better.')
  parser.add_argument('--fname_append', type=str, default='', help='str to append to routput files')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  print 'parsed parameters:'
  print json.dumps(params, indent = 2)
  main(params)
