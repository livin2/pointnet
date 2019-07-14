# -*- coding:utf-8 -*-
import os
import sys
import argparse
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ANA_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ANA_DIR)
sys.path.append(BASE_DIR)

import analysisDiff as diff

parser = argparse.ArgumentParser()
parser.add_argument('--sim',type=float, default=0.99)
FLAGS = parser.parse_args()

SIM = FLAGS.sim

SHAPE_NAMES = [line.rstrip() for line in \
    open(os.path.join(ANA_DIR, 'shape_names.txt'))]

JSO_DIR = os.path.join(BASE_DIR,'analysisAll')
if os.path.exists(JSO_DIR): shutil.rmtree(JSO_DIR)
os.mkdir(JSO_DIR)


LOG_FOUT = open(os.path.join(JSO_DIR, 'log.txt'), 'w')

if __name__=='__main__':
    # res = diff.eval_diff('cup', 'cup', SIM, JSO_DIR,LOG_FOUT)
    # diff.log_string(res.__str__(),LOG_FOUT)
    # diff.log_string('---',LOG_FOUT)

    for i in range(0,len(SHAPE_NAMES)):
        for j in range(i, len(SHAPE_NAMES)):
                if(i==j):continue
                # print (SHAPE_NAMES[i]+" "+SHAPE_NAMES[j])
                res = diff.eval_diff(SHAPE_NAMES[i], SHAPE_NAMES[j], SIM, JSO_DIR,LOG_FOUT)
                diff.log_string(res.__str__(), LOG_FOUT)
                diff.log_string('---',LOG_FOUT)

