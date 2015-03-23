import numpy as np
#import gnumpy as gp
gp = np
import code
import time
from numbapro 	import cuda
from imagernn.utils import merge_init_structs, initw, initwG, accumNpDicts
from imagernn.lstm_generator import LSTMGenerator
from imagernn.rnn_generator import RNNGenerator

def decodeGenerator(generator):
  if generator == 'lstm':
    return LSTMGenerator
  if generator == 'rnn':
    return RNNGenerator
  else:
    raise Exception('generator %s is not yet supported' % (base_generator_str,))

class GenericBatchGenerator:
  """ 
  Base batch generator class. 
  This class is aware of the fact that we are generating
  sentences from images.
  """

  @staticmethod
  def init(params, misc):

    # inputs
    image_encoding_size = params.get('image_encoding_size', 128)
    word_encoding_size = params.get('word_encoding_size', 128)
    hidden_size = params.get('hidden_size', 128)
    generator = params.get('generator', 'lstm')
    mode = params.get('mode', 'CPU')
    vocabulary_size = len(misc['wordtoix'])
    output_size = len(misc['ixtoword']) # these should match though
    image_size = 4096 # size of CNN vectors hardcoded here

    if generator == 'lstm':
      assert image_encoding_size == word_encoding_size, 'this implementation does not support different sizes for these parameters'

    # initialize the encoder models
    model = {}
    model['We'] = initw(image_size, image_encoding_size) # image encoder
    model['be'] = np.zeros((1,image_encoding_size))
    model['Ws'] = initw(vocabulary_size, word_encoding_size) # word encoder
		
    update = ['We', 'be', 'Ws']
    regularize = ['We', 'Ws']
    init_struct = { 'model' : model, 'update' : update, 'regularize' : regularize}

    # descend into the specific Generator and initialize it
    Generator = decodeGenerator(generator)
    generator_init_struct = Generator.init(word_encoding_size, hidden_size, output_size, mode)
    merge_init_structs(init_struct, generator_init_struct)
    return init_struct

  @staticmethod
  def forward(batch, model, params, misc, predict_mode = False):
    """ iterates over items in the batch and calls generators on them """
    mode = params.get('mode', 'CPU')

    # we do the encoding here across all images/words in batch in single matrix
    # multiplies to gain efficiency. The RNNs are then called individually
    # in for loop on per-image-sentence pair and all they are concerned about is
    # taking single matrix of vectors and doing the forward/backward pass without
    # knowing anything about images, sentences or anything of that sort.

    # encode all images
    # concatenate as rows. If N is number of image-sentence pairs,
    # F will be N x image_size
    #t0 = time.time()
    if mode == 'CPU':
    	F = np.row_stack(x['image']['feat'] for x in batch)
    else:
		F = gp.garray(np.row_stack(x['image']['feat'] for x in batch))
    We = model['We']
    be = model['be']
    Xe = F.dot(We) + be # Xe becomes N x image_encoding_size
    
    #dt = time.time() - t0
    #print dt

    # decode the generator we wish to use
    generator_str = params.get('generator', 'lstm') 
    Generator = decodeGenerator(generator_str)

    # encode all words in all sentences (which exist in our vocab)
    wordtoix = misc['wordtoix']
    Ws = model['Ws']
    gen_caches = []
    Ys = [] # outputs
    t0 = time.time()
    for i,x in enumerate(batch):
      # take all words in this sentence and pluck out their word vectors
      # from Ws. Then arrange them in a single matrix Xs
      # Note that we are setting the start token as first vector
      # and then all the words afterwards. And start token is the first row of Ws
      ix = [0] + [ wordtoix[w] for w in x['sentence']['tokens'] if w in wordtoix ]

      if mode == 'CPU':
        Xs = np.row_stack([Ws[j, :] for j in ix])
      else:
        Xs = gp.concatenate( [Ws[j:j+1, :] for j in ix])

      Xi = Xe[i:i+1,:]
      #dt = time.time() - t0
      #print dt
      # print Xi.shape, Xs.shape
    
      # forward prop through the RNN
      #t0 = time.time()
      gen_Y, gen_cache = Generator.forward(Xi, Xs, model, params, predict_mode = predict_mode)

      gen_caches.append((ix, gen_cache))
      Ys.append(gen_Y)
    
    dt = time.time() - t0
    print 'Forward pass time consumed %.4f' %(dt)

    # back up information we need for efficient backprop
    cache = {}
    if not predict_mode:
      # ok we need cache as well because we'll do backward pass
      cache['gen_caches'] = gen_caches
      cache['Xe'] = Xe
      cache['Ws_shape'] = Ws.shape
      cache['F'] = F
      cache['generator_str'] = generator_str

    return Ys, cache
    
  @staticmethod
  def backward(dY, cache):
    Xe = cache['Xe']
    generator_str = cache['generator_str']
    dWs = np.zeros(cache['Ws_shape'])
    gen_caches = cache['gen_caches']
    F = cache['F']
    dXe = np.zeros(Xe.shape)

    Generator = decodeGenerator(generator_str)
    dmmy, gen_cache = gen_caches[0]
    g_WLSTM = cuda.to_device(np.asfortranarray(gen_cache['WLSTM']))
    # backprop each item in the batch
    grads = {}
    dt1 = 0; dt2 =0
    t0 = time.time()
    for i in xrange(len(gen_caches)):
      t1 = time.time()
      ix, gen_cache = gen_caches[i] # unpack
      local_grads = Generator.backward(dY[i], gen_cache,g_WLSTM)
      dt1 += time.time() - t1
      
      t2 = time.time()
      dXs = local_grads['dXs'] # intercept the gradients wrt Xi and Xs
      del local_grads['dXs']
      dXi = local_grads['dXi']
      del local_grads['dXi']
      accumNpDicts(grads, local_grads) # add up the gradients wrt model parameters
      # now backprop from dXs to the image vector and word vectors
      dXe[i,:] += dXi # image vector
      for n,j in enumerate(ix): # and now all the other words
        dWs[j,:] += dXs[n,:]
      
      dt2 += time.time() - t2
      
      #dt = time.time() - t0
      #print 'BP :%0.4f' %(dt)

    dt = time.time() - t0
    print 'Backward Pass:%0.4f Others :%0.4f' %(dt1, dt2)
    t0 = time.time()
    # finally backprop into the image encoder
    dWe = F.transpose().dot(dXe)
    dbe = np.sum(dXe, axis=0, keepdims = True)
    
    dt = time.time() - t0
    print 'MMult :%0.4f' %(dt)
    t0 = time.time()

    accumNpDicts(grads, { 'We':dWe, 'be':dbe, 'Ws':dWs })
    dt = time.time() - t0
    print 'accum 2:%0.4f' %(dt)
    t0 = time.time()
    return grads

  @staticmethod
  def predict(batch, model, params, **kwparams):
    """ some code duplication here with forward pass, but I think we want the freedom in future """
    F = np.row_stack(x['image']['feat'] for x in batch) 
    We = model['We']
    be = model['be']
    Xe = F.dot(We) + be # Xe becomes N x image_encoding_size
    generator_str = params['generator']
    Generator = decodeGenerator(generator_str)
    Ys = []
    for i,x in enumerate(batch):
      gen_Y = Generator.predict(Xe[i, :], model, model['Ws'], params, **kwparams)
      Ys.append(gen_Y)
    return Ys


