"""
reading and writing data
"""
# pylint: disable=E1101
from __future__ import absolute_import, print_function, division
import os
import numpy as np
import tables
import matplotlib.pyplot as plt
from .. import NcsFile, DefaultFilter, i16File

from scipy.io import loadmat

SAMPLES_PER_REC = 512
DEFAULT_MAT_SR = 24000
DEFAULT_MAT_VOLT_FACTOR = 1

def read_matfile(fname):
    """
    read data from a matfile
    """
    data = loadmat(fname)

    try:
        sr = data['sr'].ravel()[0]
        insert = 'stored'
    except KeyError:
        sr = DEFAULT_MAT_SR
        insert = 'default'

    print('Using ' + insert + ' sampling rate ({} kHz)'.format(sr/1000.))
    ts = 1/sr
    fdata = data['data'].ravel() * DEFAULT_MAT_VOLT_FACTOR
    atimes = np.linspace(0, fdata.shape[0]/(sr/1000), fdata.shape[0])
    print(atimes.shape, fdata.shape)
    print(ts)

    return fdata, atimes, ts

def read_i16file(fname):
    """
    read data from an i16 file
    """
    sr = 20000;
    adu2uv = 1250000 / (32768 * 196)  # Intan chip range is 1.25 V per 32768 A/D units with 196 gain.

    fdata = np.fromfile(fname, np.int16, int(20000*60*10))
    
    print('Using default sampling rate ({} kHz)'.format(sr/1000.))
    ts = 1/sr
    atimes = np.linspace(0, fdata.shape[0]/(sr), fdata.shape[0])

    fdata = fdata * (adu2uv * 10**3)
    plt.plot(atimes, fdata)
    plt.axis([2, 3, -100000, 20000])  
    plt.show()  

    print(atimes.shape, fdata.shape)
    print(ts)

    return fdata, atimes, ts


class ExtractNcsFile(object):
    """
    reads data from ncs file
    """

    def __init__(self, fname, ref_fname=None):
        self.fname = fname
        self.ncs_file = NcsFile(fname) # class. represents ncs files, allows to read data and time (defaults.nlxio)
        self.ref_file = ref_fname
        if ref_fname is not None:
            self.ref_file = NcsFile(ref_fname)

        stepus = self.ncs_file.timestep * 1e6

        self.timerange = np.arange(0,
                                   SAMPLES_PER_REC * stepus,
                                   stepus)

        self.filter = DefaultFilter(self.ncs_file.timestep) # class. Simple filters for spike extraction

    def read(self, start, stop):
        """
        read data from an ncs file
        """
        data, times = self.ncs_file.read(start, stop, 'both')
        fdata = np.array(data).astype(np.float32)
        fdata *= (1e6 * self.ncs_file.header['ADBitVolts'])

        if self.ref_file is not None:
            print('Reading reference data from {}'.
                format(self.ref_file.filename))
            ref_data = self.ref_file.read(start, stop, 'data')
            fref_data = np.array(ref_data).astype(np.float32)
            fref_data *= 1e6 * self.ref_file.header['ADBitVolts']
            fdata -= fref_data

        expected_length = round((fdata.shape[0] - SAMPLES_PER_REC) *
                                (self.ncs_file.timestep * 1e6))

        err = expected_length - times[-1] + times[0]
        if err != 0:
            print("Timestep mismatch in {}"
                  " between records {} and {}: {:.1f} ms"
                  .format(self.fname, start, stop, err/1e3))

        atimes = np.hstack([t + self.timerange for t in times])/1e3
        # MUST NOT USE dictionaries here, because they would persist in memory
        return (fdata, atimes, self.ncs_file.timestep)

class Extracti16File(object):
    """
    reads data from i16 file
    """

    def __init__(self, fname, ref_fname=None):
        self.fname = fname
        self.i16_file = i16File(fname) # class. represents i16 files, allows to read data and time (defaults.nlxio)
        self.ref_file = ref_fname

        stepus = self.ncs_file.timestep * 1e6

        self.timerange = np.arange(0,
                                   SAMPLES_PER_REC * stepus,
                                   stepus)

        self.filter = DefaultFilter(self.ncs_file.timestep) # class. Simple filters for spike extraction

    def read(self, start, stop):
        """
        read data from an i16 file
        """
        data, times = self.i16_file.read(start, stop, 'both')
        fdata = np.array(data).astype(np.float32)
        # do ad conversion here later

        expected_length = round((fdata.shape[0] - SAMPLES_PER_REC) *
                                (self.ncs_file.timestep * 1e6))

        err = expected_length - times[-1] + times[0]
        if err != 0:
            print("Timestep mismatch in {}"
                  " between records {} and {}: {:.1f} ms"
                  .format(self.fname, start, stop, err/1e3))

        atimes = np.hstack([t + self.timerange for t in times])/1e3
        # MUST NOT USE dictionaries here, because they would persist in memory
        return (fdata, atimes, self.ncs_file.timestep)

class OutFile(object):
    """
    write out file to hdf5 tables
    """
    def __init__(self, name, fname, spoints=64, destination=''):

        dirname = os.path.join(destination, name)
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        fname = os.path.join(dirname, fname)
        f = tables.open_file(fname, 'w')
        f.create_group('/', 'pos', 'positive spikes')
        f.create_group('/', 'neg', 'negative spikes')

        for sign in ('pos', 'neg'):
            f.create_earray('/' + sign, 'spikes',
                           tables.Float32Atom(), (0, spoints))
            f.create_earray('/' + sign, 'times', tables.FloatAtom(), (0,))

        f.create_earray('/', 'thr', tables.FloatAtom(), (0, 3))

        self.f = f
        print('Initialized ' + fname)

    def write(self, data):
        r = self.f.root
        posspikes = data[0][0]
        postimes = data[0][1]
        negspikes = data[1][0]
        negtimes = data[1][1]

        if len(posspikes):
            r.pos.spikes.append(posspikes)
            r.pos.times.append(postimes)
        if len(negspikes):
            r.neg.spikes.append(negspikes)
            r.neg.times.append(negtimes)

        # threshold data
        r.thr.append(data[2])

        self.f.flush()

    def close(self):
        self.f.close()
