#!/usr/bin/python

from struct import unpack
import os
import numpy as np

class picsom_bin_data:

  def __init__(self, path) :
    self._path = path
    self._fp = open(path)
    self._header = self._fp.read(64)
    # print self._header[0:4]
    if self._header[0:4] != "PSBD" :
      raise Exception('Not a picsom_bin_file "%s"' % self._path)
    hdr = unpack('if7Q', self._header)
    self._version  = hdr[1]
    self._hsize    = hdr[2]
    self._rlength  = hdr[3]
    self._vdim     = hdr[4]
    self._format   = hdr[5]
    self._nobjects = (os.stat(self._path).st_size-self._hsize)/self._rlength

  def __del__(self) :
    self._fp.close()
    pass

  def vdim(self) :
    return self._vdim

  def nobjects(self) :
    return self._nobjects
  
  def get_float(self, i) :
    if (i<0 or i>=self._nobjects) :
      raise Exception('Index %i exceeds size of "%s"' % (i, self._path))
    self._fp.seek(self._hsize+i*self._rlength)
    vec = self._fp.read(self._rlength)
    return list(unpack('%if' % self._vdim, vec))
  
  def get_float_list(self, iL) :
    if iL == -1 :
	  iL = xrange(self._nobjects)
          vec = np.fromfile(self._fp, dtype = np.float32, count=self._nobjects*self._vdim )
          vec = vec.reshape(self._nobjects,self._vdim)
	  print vec.shape
    else:
	
      vec = [[]]*len(iL)
      for idx, i in enumerate(iL):
        if (i<0 or i>=self._nobjects) :
          raise Exception('Index %i exceeds size of "%s"' % (i, self._path))
        self._fp.seek(self._hsize+i*self._rlength)
        vec[idx] = list(unpack('%if' % self._vdim, self._fp.read(self._rlength)))
    return vec

if __name__ == "__main__":
  d = picsom_bin_data("/proj/mediaind/picsom/databases/COCO/features/c_in14_o6_fc6_d_a.bin")
  print d.vdim()
  print d.nobjects()
  print d.get_float_list(-1)[1][0:10]
  #print d.get_float(1)[0:10]
#  print d.get_float(10)

