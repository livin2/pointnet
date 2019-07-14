# -*- coding:utf-8 -*-
import os
import sys
import argparse
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ANA_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ANA_DIR)
sys.path.append(os.path.join(ANA_DIR, 'diff_species'))

import analysisDiff as diff

parser = argparse.ArgumentParser()
parser.add_argument('--sim',type=float, default=0.99)
FLAGS = parser.parse_args()

SIM = FLAGS.sim

SHAPE_NAMES = [line.rstrip() for line in \
    open(os.path.join(ANA_DIR, 'shape_names.txt'))]

JSO_DIR = os.path.join(BASE_DIR,'analysisV2out')
if os.path.exists(JSO_DIR): shutil.rmtree(JSO_DIR)
os.mkdir(JSO_DIR)


LOG_FOUT = open(os.path.join(JSO_DIR, 'log.txt'), 'w')
REP_FOUT = open(os.path.join(JSO_DIR, 'report.txt'), 'w')

if __name__=='__main__':
    # res = diff.eval_diff('cup', 'cup', SIM, JSO_DIR,LOG_FOUT)
    # diff.log_string(res.__str__(),LOG_FOUT)
    # diff.log_string('---',LOG_FOUT)

    for na in SHAPE_NAMES:
        res = diff.eval_diff(na, na, SIM, JSO_DIR,LOG_FOUT)
        diff.log_string(res.__str__(), LOG_FOUT)
        diff.log_string('---',LOG_FOUT)
        REP_FOUT.write(na+':'+round(res,3).__str__()+'\n')
        REP_FOUT.flush()
    LOG_FOUT.close()
    REP_FOUT.close()

