from __future__ import division, print_function, absolute_import
import os
from argparse import ArgumentParser, FileType
from .mp_extract import mp_extract
from .. import NcsFile


def get_nrecs(filename):
    fid = NcsFile(filename)
    return fid.num_recs


def main():
    """standard main function"""
    # standard options
    nWorkers = 8 # was 8
    blocksize = 10000

    parser = ArgumentParser(prog='css-extract',
                            description='spike extraction from .ncs files',
                            epilog='Johannes Niediek (jonied@posteo.de)')
    parser.add_argument('--files', nargs='+',
                        help='.ncs files to be extracted')
    parser.add_argument('--start', type=int,
                        help='start index for extraction')
    parser.add_argument('--stop', type=int,
                        help='stop index for extraction')
    parser.add_argument('--jobs', nargs=1,
                        help='job file contains one filename per row')
    parser.add_argument('--matfile', nargs=1,
                        help='extract data from a matlab file')
    parser.add_argument('--i16file', nargs=1,
                        help='extract data from a flat binary file')
    parser.add_argument('--i16filesr', nargs=1,
                        help='sampling rate for flat binary file (requires i16file)')
    parser.add_argument('--destination', nargs=1,
                        help='folder where spikes should be saved')
    parser.add_argument('--refscheme', nargs=1, type=FileType(mode='r'),
                        help='scheme for re-referencing')
    args = parser.parse_args()

    if (args.files is None) and (args.matfile is None) and\
            (args.jobs is None) and (args.i16file is None):
        parser.print_help()
        print('Supply either files or jobs or matfile or binary file.')
        return

    if args.destination is not None:
        destination = args.destination[0]
    else:
        destination = ''

    # special case for a matlab file
    if args.matfile is not None:
        jname = os.path.splitext(os.path.basename(args.matfile[0]))[0]
        jobs = [{'name': jname,
                 'filename': args.matfile[0],
                 'is_matfile': True,
                 'count': 0,
                 'destination': destination}]
        mp_extract(jobs, 1)
        return

    # special case for I16 file
    #if args.i16file is not None:
        #jname = os.path.splitext(os.path.basename(args.i16file[0]))[0]
        #jobs = [{'name': jname,
                 #'filename': args.i16file[0],
                 #'is_i16file': True,
                 #'count': 0,
                 #'destination': destination}]
        #mp_extract(jobs, 1)
        #return

    # generate file list
    if args.jobs: # if we give a text file of mutiple files
        with open(args.jobs[0], 'r') as f:
            files = [a.strip() for a in f.readlines()]
        f.close()
        print('Read jobs from ' + args.jobs[0])
    elif args.i16file:
        files = args.i16file
    else:
        files = args.files

    print(files)

    if files[0] is None:
        print('Specify files!')
        return

    # construct the jobs
    jobs = []

    references = None
    if args.refscheme:
        import csv
        reader = csv.reader(args.refscheme[0], delimiter=';')
        references = {line[0]: line[1] for line in reader}

    for f in files:
        if args.start:
            start = args.start
        else:
            start = 0
        nrecs = 1000000 # cheat. later look up length of file here if not ncs file
        #nrecs = get_nrecs(f) # get length of file
        if args.stop:
            stop = min(args.stop, nrecs)
        else:
            stop = nrecs

        if stop % blocksize > blocksize/2: # dont try to read past end of file
            laststart = stop-blocksize
        else:
            laststart = stop

        if args.i16filesr:
            i16filesr = args.i16filesr
        else:
            i16filesr = 20000

        starts = range(start, laststart, blocksize)
        stops = starts[1:] + [stop]
        name = os.path.splitext(os.path.basename(f))[0]
        if references is not None:
            reference = references[f]
            print('{} (re-referenced to {})'.format(f, reference))

        else:
            reference = None
            print(name)

        for i in range(len(starts)):
            jdict = {'name': name,
                     'filename': f,
                     'start': starts[i],
                     'stop': stops[i],
                     'count': i,
                     'destination': destination,
                     'reference': reference,
                     'i16filesr': i16filesr}

            jobs.append(jdict)


    mp_extract(jobs, nWorkers) # recursively go through all these blocks
