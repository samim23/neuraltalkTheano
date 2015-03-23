import argparse
import json
import time
import datetime
import numpy as np
import code
import socket
import os
import cPickle as pickle
import math
import theano
from theano import config
import theano.tensor as tensor

from imagernn.data_provider import getDataProvider
from imagernn.solver import Solver
from imagernn.imagernn_utils import decodeGenerator, eval_split, eval_split_theano

def main(params):

  # load the checkpoint
  result_struct = params['checkpoint_path']
  max_images = params['max_images']

  print 'loading result data %s' % (result_struct, )
  resultDb = json.load(open(result_struct,'r'))

  checkpoint_params = resultDb['checkpoint_params']

  dataset = checkpoint_params['dataset']
  dump_folder = params['dump_folder']

  if 'image_feat_size' not in  checkpoint_params:
    checkpoint_params['image_feat_size'] = 4096 

  if dump_folder:
    print 'creating dump folder ' + dump_folder
    os.system('mkdir -p ' + dump_folder)
    
  # fetch the data provider
  dp = getDataProvider(checkpoint_params)
  
  fTrn = open('dBSentFile', 'w')
  fTst = open('queryFile', 'w')
  n = 0
  for img in dp.iterImages(split = 'train', max_images = max_images):
    n += 1
    print 'image %d/%d:' % (n, max_images)
    fTrn.writelines("%s\n"% ' '.join(x['tokens']) for x in img['sentences'])

  fTst.writelines("%s\n"% x['candidate']['text'] for x in resultDb['imgblobs'])

  
  fNNRes = open('Brute_SearchResult_FULL.txt')
  tups = re.findall(pattern,f.read())
  fNNRes.close()

  for t in tups:
    trnIdx = int(t[1])
    tstIdx = int(t[0])
    imgIdx = floor(trnIdx / 5)
    senIdx = trnIdx - imgIdx * 5

    nnSent = 
    


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('checkpoint_path', type=str, help='the input checkpoint')
  parser.add_argument('--result_struct_filename', type=str, default='result_struct.json', help='filename of the result struct to save')
  parser.add_argument('-m', '--max_images', type=int, default=-1, help='max images to use')
  parser.add_argument('-d', '--dump_folder', type=str, default="", help='dump the relevant images to a separate folder with this name?')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  print 'parsed parameters:'
  print json.dumps(params, indent = 2)
  main(params)
