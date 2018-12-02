# -*- coding:utf-8 -*- 
import numpy as np
import os
import shutil
import json
import argparse
import codecs
import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PAR_DIR  = os.path.dirname(os.path.dirname(BASE_DIR))
PICS_DIR = os.path.join(PAR_DIR,'result/final')
RLG_DIR = os.path.join(PAR_DIR,'result/rot_log')

parser = argparse.ArgumentParser()
parser.add_argument('--obj1', default='flower_pot')
parser.add_argument('--obj2', default='plant')
parser.add_argument('--cppic', action='store_true')
parser.add_argument('--sim',type=float, default=0.99)
FLAGS = parser.parse_args()

OBJ1 = FLAGS.obj1
OBJ2 = FLAGS.obj2

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return obj.item()
        if isinstance(obj, np.float64):
            return obj.item()
        return json.JSONEncoder.default(self, obj)

#对两矩阵对应向量求余弦相似度取平均
def cosine_distance(matrix1, matrix2):
    matrix1_matrix2 = np.dot(matrix1, matrix2.transpose())                                                
    matrix1_norm = np.sqrt(np.multiply(matrix1, matrix1).sum(axis=1))
    matrix1_norm = matrix1_norm[:, np.newaxis]
    matrix2_norm = np.sqrt(np.multiply(matrix2, matrix2).sum(axis=1))
    matrix2_norm = matrix2_norm[:, np.newaxis]
    cosine_distance = np.divide(matrix1_matrix2, np.dot(matrix1_norm, matrix2_norm.transpose()))
    return np.average(cosine_distance.diagonal())

#根据余弦相似度分类
def select_hgihsim3X3(obj1,obj2,sim):
    if sim>=1:
        raise Exception('expect sim <1')
    Tnet1 = dict(np.load('%s.npz'%(obj1)))
    Tnet2 = dict(np.load('%s.npz' % (obj2)))

    print ('obj1 num:'+len(Tnet1).__str__())
    print ('obj2 num:'+len(Tnet2).__str__())

    selected = []
    for key1 in Tnet1:
        for key2 in Tnet2:
            avg_similarity = cosine_distance(Tnet1[key1], Tnet2[key2])
            if (avg_similarity > sim):
                selected.append([{'sim':avg_similarity,'%s'%OBJ1:Tnet1[key1],'%s'%OBJ2:Tnet2[key2]}])
    return len(Tnet1),len(Tnet2),selected


if __name__=='__main__':
    obj1 = os.path.join(RLG_DIR,OBJ1)
    obj2 = os.path.join(RLG_DIR, OBJ2)
    num1,num2,selected = select_hgihsim3X3(obj1,obj2,FLAGS.sim)

    print ('similar 3X3 NUM:'+len(selected).__str__())
    json_file = "%d%%%d-%s-%s-%f.json" % (len(selected),num1*num2,OBJ1,OBJ2,FLAGS.sim)
    print (json_file)
    json_file = os.path.join(BASE_DIR, json_file)
    json.dump(selected, codecs.open(json_file, 'w', encoding='utf-8'), sort_keys=True, indent=4,cls=NumpyEncoder)

