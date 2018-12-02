# -*- coding:utf-8 -*- 
import numpy as np
import os
import shutil
import json
import argparse
import codecs
import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#对两矩阵对应向量求余弦相似度取平均
def cosine_distance(matrix1, matrix2):
    matrix1_matrix2 = np.dot(matrix1, matrix2.transpose())                                                
    matrix1_norm = np.sqrt(np.multiply(matrix1, matrix1).sum(axis=1))
    matrix1_norm = matrix1_norm[:, np.newaxis]
    matrix2_norm = np.sqrt(np.multiply(matrix2, matrix2).sum(axis=1))
    matrix2_norm = matrix2_norm[:, np.newaxis]
    cosine_distance = np.divide(matrix1_matrix2, np.dot(matrix1_norm, matrix2_norm.transpose()))
    return cosine_distance
    # return np.average(cosine_distance.diagonal())

if __name__=='__main__':
    mt1 = np.array([
                [
                    9.225212097167969,
                    -2.6717541217803955,
                    0.6590409874916077
                ],
                [
                    -0.0335657000541687,
                    14.30512809753418,
                    0.47776874899864197
                ],
                [
                    -1.9491041898727417,
                    0.91061931848526,
                    8.537610054016113
                ]
            ])

    mt2 = np.array([
                [
                    5.283310413360596,
                    -1.0944466590881348,
                    0.24036487936973572
                ],
                [
                    0.6669061183929443,
                    8.815923690795898,
                    0.22688938677310944
                ],
                [
                    -0.48021259903907776,
                    0.17859430611133575,
                    5.2169294357299805
                ]
            ])

    # print ((np.multiply(mt1[0], mt1[0])))
    # m1s = np.multiply(mt1[0], mt1[0]).sum()
    # print (m1s)
    # m1n = np.sqrt(m1s)
    # print (m1n)
    # print ('\n')
    #
    # print ((np.multiply(mt2[0], mt2[0])))
    # m2s = np.multiply(mt2[0], mt2[0]).sum()
    # print (m2s)
    # m2n = np.sqrt(m2s)
    # print (m2n)
    # print ('\n')
    #
    # mdot = np.dot(mt1[0], mt2[0].transpose())
    # print (mdot)
    #
    # ccdd = np.divide(mdot, np.multiply(m1n,m2n))
    # print (ccdd)


    cd = cosine_distance(mt1,mt2)
    print (cd)
    print (cd.diagonal())
    print (np.average(cd.diagonal()))

