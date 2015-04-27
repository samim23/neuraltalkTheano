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
from tsne import bh_sne 
from bokeh.plotting import ColumnDataSource, figure, output_file, show, save, VBox, HBox
from bokeh.models import HoverTool

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

  z1 = bh_sne(checkpoint['model']['Wemb'].astype('float64')); 
  z2 = bh_sne(checkpoint['model']['Wd'].T.astype('float64')); 
  
  filepath = 'tsne_result_%s.p' % (params['fname_append'] ) 
  TsneL = {}
  TsneL['Wemb'] = z1
  TsneL['Wd'] = z2
  TsneL['model_file'] = checkpoint_path
  pickle.dump(TsneL, open(filepath, "wb"))

  filepath = 'scatter_Wemb_%s.html' % (params['fname_append'] ) 
  output_file(filepath)
  TOOLS="pan,wheel_zoom,box_zoom,reset,hover"

  p1 = figure(title="Word embedding Matrix rows", tools=TOOLS)
  source1 = ColumnDataSource(data=dict(x=z1[:,0], y=z1[:,1],lab = checkpoint['ixtoword'].values()))
  p1.circle(z1[:,0], z1[:,1], source=source1,line_color=None)
  hover1 = p1.select(dict(type=HoverTool))
  hover1.tooltips = [("(x,y)","($x,$y)"),("text","@lab")]

  p2 = figure(title="Word Decode Matrix columns", tools=TOOLS)
  source2 = ColumnDataSource(data=dict(x=z2[:,0], y=z2[:,1],lab = checkpoint['ixtoword'].values()))
  p2.circle(z2[:,0], z2[:,1], source=source2,line_color=None)
  hover2 = p2.select(dict(type=HoverTool))
  hover2.tooltips = [("(x,y)","($x,$y)"),("text","@lab")]

  idx = np.argsort(checkpoint['model']['bd'])
  

  source3 = ColumnDataSource(data=dict(x=range(len(checkpoint['model']['bd'])), y=checkpoint['model']['bd'][idx[::-1]],lab = [checkpoint['ixtoword'][i] for i in idx[::-1]]))
  p3 = figure(title="Bias Vector", tools=TOOLS)
  p3.circle(range(len(checkpoint['model']['bd'])),checkpoint['model']['bd'][idx[::-1]],source = source3)
  hover3 = p3.select(dict(type=HoverTool))
  hover3.tooltips = [("(x,y)","($x,$y)"),("text","@lab")]

  save(VBox(p1, HBox(p2,p3)))


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
