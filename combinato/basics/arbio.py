# -*- coding: utf-8 -*-
"""
Basic i/o definitions for Neuralynx files
"""

from __future__ import print_function, division, absolute_import
from os import stat
from datetime import datetime
import re
import numpy as np
# pylint: disable=E1101

i16_type = np.dtype([('data', 'i2')])

class i16File(object):
    """
    represents i16 files, allows to read data and time
    """
    def __init__(self, filename, sr):
        self.file = None
        self.filename = filename
        self.num_recs = 1000000 #ncs_num_recs(filename)
        #self.header = ncs_info(filename)
        self.file = open(filename, 'rb')
        self.timestep = 1 / (sr * 1e6) # timestep in ms? guess by arb
        self.sr = sr

    def __del__(self):
        if self.file is not None:
            self.file.close()

    def read(self, start=0, stop=None, mode='data'):
        """
        read data, timestamps, or info fields from ncs file
        """
        if stop > start:
            length = stop - start
        else:
            length = 1
        if start + length > self.num_recs + 1:
            raise IOError("Request to read beyond EOF,"
                          "filename %s, start %i, stop %i" %
                          (self.filename, start, stop))
        else:
            self.file.seek(start)
            data = self.file.read(length*2)
            print(length)
            array_length = int(len(data))
            print(array_length)
            #print(array_length)
            data = np.ndarray(length, 'i2', data)
            #array_data = np.ndarray(array_length, np.int16, data)
            atimes = np.linspace(start/self.sr, stop/self.sr, data.shape[0]) # generate timestamps            
            if mode == 'both':
                return (data,atimes)
            elif mode == 'timestamp':
                return (atimes)
            elif mode == 'data':
                return (data)



