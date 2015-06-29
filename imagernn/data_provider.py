import json

import os
import random
import h5py
import scipy.io
import codecs
import numpy as np
import theano
from collections import defaultdict
from picsom_bin_data import picsom_bin_data

class BasicDataProvider:
  def __init__(self, params):
    dataset = params.get('dataset', 'coco')
    feature_file = params.get('feature_file', 'vgg_feats.mat')
    data_file = params.get('data_file', 'dataset.json')
    mat_new_ver = params.get('mat_new_ver', -1)
    print 'Initializing data provider for dataset %s...' % (dataset, )
    self.hdf5Flag = 0 #Flag indicating whether the dataset is an HDF5 File.
                 #Large HDF5 files are stored (by Vik) as one image 
                 #  per row, going against the conventions of the other
                 # storage formats

    # !assumptions on folder structure
    self.dataset_root = os.path.join('data', dataset)
    self.image_root = os.path.join('data', dataset, 'imgs')

    # load the dataset into memory
    dataset_path = os.path.join(self.dataset_root, data_file)
    print 'BasicDataProvider: reading %s' % (dataset_path, )
    self.dataset = json.load(open(dataset_path, 'r'))

    # load the image features into memory
    features_path = os.path.join(self.dataset_root, feature_file)
    print 'BasicDataProvider: reading %s' % (features_path, )
    
    if feature_file.rsplit('.',1)[1] == 'mat':
        if mat_new_ver == 1:
            features_struct = h5py.File(features_path)
            self.features = np.array(features_struct[features_struct.keys()[0]],dtype=theano.config.floatX)
        else:
            features_struct = scipy.io.loadmat(features_path)
            self.features = features_struct['feats']
    #The condition below makes consuming HDF5 features easy
    #   This is what I (Vik) use for features extracted in an unsupervised
    #   manner
    elif feature_file.rsplit('.',1)[1] == 'hdf5':
        self.hdf5Flag = 1
        features_struct = h5py.File(features_path)
        self.features = features_struct['features'] #The dataset in the HDF5 file is named 'features'
    elif feature_file.rsplit('.',1)[1] == 'bin':
        features_struct = picsom_bin_data(features_path) 
        self.features = np.array(features_struct.get_float_list(-1)).T.astype(theano.config.floatX) 
		# this is a 4096 x N numpy array of features
        print "Working on Bin file now"                                        
    elif feature_file.rsplit('.',1)[1] == 'txt':
        #This is for feature concatenation.
        # NOTE: Assuming same order of features in all the files listed in the txt file
        feat_Flist = open(features_path, 'r').read().splitlines()
        
        feat_list = []
        
        for f in feat_Flist:
            f_struct = picsom_bin_data(os.path.join(self.dataset_root,f)) 
            feat_list.append(np.array(f_struct.get_float_list(-1)).T.astype(theano.config.floatX))
            print feat_list[-1].shape
		    # this is a 4096 x N numpy array of features
        self.features = np.concatenate(feat_list, axis=0)
        print "Combined all the features. Final size is %d %d"%(self.features.shape[0],self.features.shape[1])
    
    if self.hdf5Flag == 1:
        #Because the HDF5 file is currently stored as one feature per row 
        self.img_feat_size = self.features.shape[1]
    else: 
        self.img_feat_size = self.features.shape[0]
        

    self.aux_pres = 0
    self.aux_inp_size = 0
    if params.get('en_aux_inp',0):
        # Load Auxillary input file, one vec per image
        # NOTE: Assuming same order as feature file
        f_struct = picsom_bin_data(os.path.join(self.dataset_root,params['aux_inp_file'])) 
        self.aux_inputs = np.array(f_struct.get_float_list(-1)).T.astype(theano.config.floatX) 
        self.aux_pres = 1
        self.aux_inp_size = self.aux_inputs.shape[0]


    # group images by their train/val/test split into a dictionary -> list structure
    self.split = defaultdict(list)
    for img in self.dataset['images']:
      self.split[img['split']].append(img)
      if img['split'] != 'train':
        self.split['allval'].append(img)
    
    # Build tables for length based sampling
    lenHist = defaultdict(int)
    self.lenMap = defaultdict(list)
    if(dataset == 'coco'):
        self.min_len = 7
        self.max_len = 27
        for iid, img in enumerate(self.split['train']): 
          for sid, sent in enumerate(img['sentences']):
            ix = max(min(len(sent['tokens']),self.max_len),self.min_len)
            lenHist[ix] += 1
            self.lenMap[ix].append((iid,sid))
    else:
        raise ValueError('ERROR: Dont know how to do len splitting for this dataset')

    self.lenCdist = np.cumsum(lenHist.values())

    
  # "PRIVATE" FUNCTIONS
  # in future we may want to create copies here so that we don't touch the 
  # data provider class data, but for now lets do the simple thing and 
  # just return raw internal img sent structs. This also has the advantage
  # that the driver could store various useful caching stuff in these structs
  # and they will be returned in the future with the cache present
  def _getImage(self, img):
    """ create an image structure for the driver """

    # lazily fill in some attributes
    if not 'local_file_path' in img: img['local_file_path'] = os.path.join(self.image_root, img['filename'])
    if not 'feat' in img: # also fill in the features
      feature_index = img['imgid'] # NOTE: imgid is an integer, and it indexes into features
      if self.hdf5Flag == 1:  #If you're reading from an HDF5 File
          img['feat'] = self.features[feature_index, :]
      else: 
          img['feat'] = self.features[:,feature_index]
      if self.aux_pres:
        img['aux_inp'] = self.aux_inputs[:,feature_index]
    return img

  def _getSentence(self, sent):
    """ create a sentence structure for the driver """
    # NOOP for now
    return sent

  # PUBLIC FUNCTIONS

  def getSplitSize(self, split, ofwhat = 'sentences'):
    """ return size of a split, either number of sentences or number of images """
    if ofwhat == 'sentences': 
      return sum(len(img['sentences']) for img in self.split[split])
    else: # assume images
      return len(self.split[split])

  def sampleImageSentencePair(self, split = 'train'):
    """ sample image sentence pair from a split """
    images = self.split[split]

    img = random.choice(images)
    sent = random.choice(img['sentences'])

    out = {}
    out['image'] = self._getImage(img)
    out['sentence'] = self._getSentence(sent)
    return out
  
  def sampleImageSentencePairByLen(self, l, split='train'):
    """ sample image sentence pair from a split """
    pair = random.choice(self.lenMap[l])

    img = self.split[split][pair[0]]
    sent = img['sentences'][pair[1]]

    out = {}
    out['image'] = self._getImage(img)
    out['sentence'] = self._getSentence(sent)
    return out
  
  def getRandBatchByLen(self,batch_size):
    """ sample image sentence pair from a split """
    
    rn = np.random.randint(0,self.lenCdist[-1])
    for l in xrange(len(self.lenCdist)):
        if rn < self.lenCdist[l]:
            break

    l += self.min_len
    batch = [self.sampleImageSentencePairByLen(l) for i in xrange(batch_size)]
    return batch,l

  def iterImageSentencePair(self, split = 'train', max_images = -1):
    for i,img in enumerate(self.split[split]):
      if max_images >= 0 and i >= max_images: break
      for sent in img['sentences']:
        out = {}
        out['image'] = self._getImage(img)
        out['sentence'] = self._getSentence(sent)
        yield out

  def iterImageSentencePairBatch(self, split = 'train', max_images = -1, max_batch_size = 100):
    batch = []
    for i,img in enumerate(self.split[split]):
      if max_images >= 0 and i >= max_images: break
      for sent in img['sentences']:
        out = {}
        out['image'] = self._getImage(img)
        out['sentence'] = self._getSentence(sent)
        batch.append(out)
        if len(batch) >= max_batch_size:
          yield batch
          batch = []
    if batch:
      yield batch

  def iterSentences(self, split = 'train'):
    for img in self.split[split]: 
      for sent in img['sentences']:
        yield self._getSentence(sent)

  def iterImages(self, split = 'train', shuffle = False, max_images = -1):
    imglist = self.split[split]
    ix = range(len(imglist))
    if shuffle:
      random.shuffle(ix)
    if max_images > 0:
      ix = ix[:min(len(ix),max_images)] # crop the list
    for i in ix:
      yield self._getImage(imglist[i])
  
def prepare_data( batch, wordtoix, maxlen=None,sentTagMap=None,ixw = None):
    """Create the matrices from the datasets.

    This pad each sequence to the same length: the lenght of the
    longest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    length.

    This swap the axis!
    """
    seqs = []
    xI = np.row_stack(x['image']['feat'] for x in batch)

    for ix,x in enumerate(batch):
      seqs.append([0] + [wordtoix[w] for w in x['sentence']['tokens'] if w in wordtoix] + [0])

    # x: a list of sentences
    lengths = [len(s) for s in seqs]

    if maxlen is not None:
        new_seqs = []
        new_labels = []
        new_lengths = []
        for l, s, y in zip(lengths, seqs, labels):
            if l < maxlen:
                new_seqs.append(s)
                new_labels.append(y)
                new_lengths.append(l)
        lengths = new_lengths
        labels = new_labels
        seqs = new_seqs

        if len(lengths) < 1:
            return None, None, None

    n_samples = len(seqs)
    maxlen = np.max(lengths)

    xW = np.zeros((maxlen, n_samples)).astype('int64')
    x_mask = np.zeros((maxlen, n_samples)).astype(theano.config.floatX)
    for idx, s in enumerate(seqs):
        xW[:lengths[idx], idx] = s
        x_mask[:lengths[idx], idx] = 1.
        if sentTagMap != None:
            for i,sw in enumerate(s):
                if sentTagMap[batch[idx]['sentence']['sentid']].get(ixw[sw],'') == 'JJ':
                    x_mask[i,idx] = 2 
                

    inp_list = [xW, xI, x_mask]

    if 'aux_inp' in batch[0]['image']:
      xAux = np.row_stack(x['image']['aux_inp'] for x in batch)
      #xAux = np.tile(xAux,[maxlen,1,1])
      inp_list.append(xAux)

    return inp_list, (np.sum(lengths) - n_samples)

def getDataProvider(params):
  """ we could intercept a special dataset and return different data providers """
  assert params['dataset'] in ['flickr8k', 'flickr30k', 'coco'], 'dataset %s unknown' % (dataset, )
  return BasicDataProvider(params)

def loadArbitraryFeatures(params, idxes):
  
  feat_all = []
  aux_all = []
  if params.get('multi_model',0) == 0:
    params['nmodels'] = 1
    
  for i in xrange(params['nmodels']):
      #----------------------------- Loop -------------------------------------#
      features_path = params['feat_file'][i] if params.get('multi_model',0) else params['feat_file']
      if features_path.rsplit('.',1)[1] == 'mat':
        features_struct = scipy.io.loadmat(features_path)
        features = features_struct['feats'][:,idxes].astype(theano.config.floatX) # this is a 4096 x N numpy array of features
      elif features_path.rsplit('.',1)[1] == 'hdf5':
        #If the file is one of Vik's HDF5 Files
        features_struct = h5py.File(features_path,'r')
        features = features_struct['feats'][idxes,:].astype(theano.config.floatX) # this is a N x 2032128 array of features
      elif features_path.rsplit('.',1)[1] == 'bin':
        features_struct = picsom_bin_data(features_path) 
        features = np.array(features_struct.get_float_list(idxes)).T.astype(theano.config.floatX) # this is a 4096 x N numpy array of features
        print "Working on Bin file now"
      elif features_path.rsplit('.',1)[1] == 'txt':
          #This is for feature concatenation.
          # NOTE: Assuming same order of features in all the files listed in the txt file
          feat_Flist = open(features_path, 'r').read().splitlines()
          feat_list = []
          for f in feat_Flist:
              f_struct = picsom_bin_data(f) 
              feat_list.append(np.array(f_struct.get_float_list(idxes)).T)
              print feat_list[-1].shape
      	  # this is a 4096 x N numpy array of features
          features = np.concatenate(feat_list, axis=0).astype(theano.config.floatX)
          print "Combined all the features. Final size is %d %d"%(features.shape[0],features.shape[1])
      
      aux_inp = []
      aux_inp_file = params['aux_inp_file'][i] if params.get('multi_model',0) else params['aux_inp_file']
      if aux_inp_file != None:
          # Load Auxillary input file, one vec per image
          # NOTE: Assuming same order as feature file
          f_struct = picsom_bin_data(aux_inp_file) 
          aux_inp = np.array(f_struct.get_float_list(idxes)).T.astype(theano.config.floatX) 
      
      feat_all.append(features)
      aux_all.append(aux_inp)

  if params.get('multi_model',0) == 0:
    return features, aux_inp
  else:
    return feat_all, aux_all

