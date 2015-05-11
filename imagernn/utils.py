from random import uniform
import numpy as np
from theano import config
from collections import OrderedDict
#import gnumpy as gp
gp = np
#from numbapro.cudalib.cublas import Blas as cublas
#import numbapro.cudalib.cublas as cublas
#blas = cublas.Blas()

def randi(N):
  """ get random integer in range [0, N) """
  return int(uniform(0, N))

def merge_init_structs(s0, s1):
  """ merge struct s1 into s0 """
  for k in s1['model']:
    assert (not k in s0['model']), 'Error: looks like parameter %s is trying to be initialized twice!' % (k, )
    s0['model'][k] = s1['model'][k] # copy over the pointer
  s0['update'].extend(s1['update'])
  s0['regularize'].extend(s1['regularize'])

def initw(n,d): # initialize matrix of this size
  magic_number = 0.1
  return (np.random.rand(n,d) * 2 - 1) * magic_number # U[-0.1, 0.1]

def initwTh(n,d,magic_number=0.1): # initialize matrix of this size
  return ((np.random.rand(n,d) * 2 - 1) * magic_number).astype(config.floatX) # U[-0.1, 0.1]

def _p(pp, name):
    return '%s_%s' % (pp, name)

def numpy_floatX(data):
    return np.asarray(data, dtype=config.floatX)

def initwG(n,d): # initialize matrix of this size
  magic_number = 0.1
  return (gp.rand(n,d) * 2 - 1) * magic_number # U[-0.1, 0.1]

def accumNpDicts(d0, d1):
  """ forall k in d0, d0 += d1 . d's are dictionaries of key -> numpy array """
  for k in d1:
    if k in d0:
      d0[k] += d1[k]
    else:
      d0[k] = d1[k]

def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)

def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params

def forwardSubRoutine(Hin,Hout, X, WLSTM,IFOG,IFOGf,C,n,d):

    for t in xrange(n):

      prev = np.zeros(d) if t == 0 else Hout[t-1]
      #tanhC_version = 1
      Hin[t,0] = 1
      Hin[t,1:1+d] = X[t]
      Hin[t,1+d:] = prev

      # compute all gate activations. dots:
      IFOG[t] = Hin[t].dot(WLSTM)
      
      IFOGf[t,:3*d] = 1.0/(1.0+np.exp(-IFOG[t,:3*d])) # sigmoids; these are the gates
      IFOGf[t,3*d:] = np.tanh(IFOG[t, 3*d:]) # tanh

      C[t] = IFOGf[t,:d] * IFOGf[t, 3*d:]
      if t > 0: C[t] += IFOGf[t,d:2*d] * C[t-1]
      
      Hout[t] = IFOGf[t,2*d:3*d] * np.tanh(C[t])
      
      #  Hout[t] = IFOGf[t,2*d:3*d] * C[t]
    return Hin, Hout, IFOG,IFOGf,C
	
#def backMultSubroutine(g_Hin, WLSTM,dIFOG, dWLSTM,dHin):    
#    blas.gemm('N','N',g_Hin.shape[0],dIFOG.shape[1],1,1.0,g_Hin, dIFOG,1.0,dWLSTM)
#    blas.gemm('N','N',dIFOG.shape[0],WLSTM.shape[1],WLSTM.shape[0],1.0,dIFOG,WLSTM,0.0,dHin)
#	
#    return dHin, dWLSTM
	
