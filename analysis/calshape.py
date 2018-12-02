import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PAR_DIR  = os.path.dirname(os.path.dirname(BASE_DIR))
RLG_DIR = os.path.join(PAR_DIR,'result/rot_log')


LOG_FOUT = open(os.path.join(BASE_DIR, 'calShape.txt'), 'w')

NUM_CLASSES = 40
SHAPE_NAMES = [line.rstrip() for line in \
    open(os.path.join(BASE_DIR, 'shape_names.txt'))]


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

if __name__=='__main__':
    for name in SHAPE_NAMES:
        filename = '%s.npz'%name
        filename = os.path.join(RLG_DIR,filename)
        arr = dict(np.load(filename))
        log_string(name +':'+len(arr).__str__())
    LOG_FOUT.close()
