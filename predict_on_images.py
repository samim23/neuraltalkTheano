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
from imagernn.utils import zipp

"""
This script is used to predict sentences for arbitrary images
"""

def main(params):

  # load the checkpoint
  if params['multi_model'] == 0:
    checkpoint_path = params['checkpoint_path']
    print 'loading checkpoint %s' % (checkpoint_path, )
    checkpoint = pickle.load(open(checkpoint_path, 'rb'))
    checkpoint_params = checkpoint['params']
    model_npy = checkpoint['model']
    checkpoint_params['use_theano'] = 1
    if 'image_feat_size' not in  checkpoint_params:
        checkpoint_params['image_feat_size'] = 4096 
    
    BatchGenerator = decodeGenerator(checkpoint_params)
    # Compile and init the theano predictor 
    BatchGenerator.prepPredictor(model_npy, checkpoint_params, params['beam_size'])
    model = BatchGenerator.model_th
  else:
    BatchGenerator = []
    model_npy = []
    modelTh = []
    checkpoint_params = []
    for i,checkpoint_path in enumerate(params['checkpoint_path']):
        checkpoint = pickle.load(open(checkpoint_path, 'rb'))
        model_npy.append(checkpoint['model'])
        checkpoint_params.append(checkpoint['params'])
        checkpoint_params[i]['use_theano'] = 1
        BatchGenerator.append(decodeGenerator(checkpoint_params[i]))
        zipp(model_npy[i],BatchGenerator[i].model_th)
        modelTh.append(BatchGenerator[i].model_th)
        modelTh[i]['comb_weight'] = 1.0/params['nmodels']
    
    BatchGenerator[0].prepMultiPredictor(modelTh,checkpoint_params,params['beam_size'],params['nmodels'])
  
  
  misc = {}
  ixtoword = checkpoint['ixtoword']
  misc['wordtoix'] = checkpoint['wordtoix']


  # output blob which we will dump to JSON for visualizing the results
  blob = {} 
  blob['params'] = params
  blob['checkpoint_params'] = checkpoint_params
  blob['imgblobs'] = []

  # load the tasks.txt file and setupe feature loading
  root_path = params['root_path']
  img_names_list = open(params['imgList'], 'r').read().splitlines()

  if len(img_names_list[0].rsplit(',')) > 1:
    img_names = [x.rsplit (',')[0] for x in img_names_list]
    idxes = [int(x.rsplit (',')[1]) for x in img_names_list]
  else:
    img_names = img_names_list
    idxes = xrange(len(img_names_list))
  
  #if checkpoint_params.get('en_aux_inp',0) and (params.get('aux_inp_file','None') == 'None'):
  #  raise ValueError('ERROR: please specify auxillary input feature using --aux_inp_file')
  #  return
  # load the features for all images
  features, aux_inp = loadArbitraryFeatures(params, idxes)

  N = len(img_names) 

  # iterate over all images and predict sentences
  print("\nUsing model run for %0.2f epochs with validation perplx at %0.3f\n" % (checkpoint['epoch'], \
    checkpoint['perplexity']))
  
  kwparams = { 'beam_size' : params['beam_size'] }
  
  jsonFname = 'result_struct_%s.json' % (params['fname_append'] ) 
  save_file = os.path.join(root_path, jsonFname)
  
  for n in xrange(N):
    print 'image %d/%d:' % (n, N)

    # encode the image
    if params['multi_model'] == 0:
        D,NN = features.shape
        img = {}
        img['feat'] = features[:, n]
        if checkpoint_params.get('en_aux_inp',0):
            img['aux_inp'] = aux_inp[:, n]
        img['local_file_path'] =img_names[n]
        # perform the work. heavy lifting happens inside
        Ys = BatchGenerator.predict([{'image':img}], model, checkpoint_params, **kwparams)
    else:
        kwparams['nmodels'] = params['nmodels']
        batch = []
        for i in xrange(params['nmodels']):
            img = {}
            img['feat'] = features[i][:, n]
            if checkpoint_params[i].get('en_aux_inp',0):
                img['aux_inp'] = aux_inp[i][:, n]
            img['local_file_path'] =img_names[n]
            batch.append({'image':img})
        Ys = BatchGenerator[0].predictMulti(batch, checkpoint_params, **kwparams)

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
    if (n%5000) == 1:
        print 'writing predictions to %s...' % (save_file, )
        json.dump(blob, open(save_file, 'w'))

  # dump result struct to file
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
  parser.add_argument('--aux_inp_file', dest='aux_inp_file', type=str, default='None', help='Is there any auxillary inputs ? If yes indicate file here')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict

  if params['aux_inp_file'] != 'None':
    params['en_aux_inp'] = 1
  else:
    params['en_aux_inp'] = 0

  print 'parsed parameters:'
  print json.dumps(params, indent = 2)
  params['multi_model'] = 0

  if params['checkpoint_path'].rsplit('.')[-1] == 'txt':
    # Dealing with multiple models now, Setup the checkpointa and feature filenames
    params['multi_model'] = 1
    model_list = open(params['checkpoint_path'],'r').read().splitlines()
    params['nmodels'] = len(model_list)
    params['checkpoint_path'] = ['']*len(model_list)
    params['feat_file'] = ['']*len(model_list)
    params['aux_inp_file'] = ['']*len(model_list)
    for i,m in enumerate(model_list):
        cauxFname = m.split(',')
        params['checkpoint_path'][i] = cauxFname[0]
        params['feat_file'][i] = cauxFname[1].lstrip(' ').rstrip(' ')
        params['aux_inp_file'][i] = cauxFname[2].lstrip(' ').rstrip(' ') if len(cauxFname) > 2 else None

    print params['checkpoint_path']


  main(params)
