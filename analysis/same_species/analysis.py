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
parser.add_argument('--obj', default='airplane')
parser.add_argument('--sim',type=float, default=0.9)
FLAGS = parser.parse_args()

OBJ_NAME = FLAGS.obj
NAME_STR1 = '%s_beforeR_label_'+OBJ_NAME+'_pred_*.jpg'
NAME_STR2 = '%s_TnetR_label_'+OBJ_NAME+'_pred_*.jpg'

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
def category_by_avg_similarity(obj_name,sim):
    if sim>=1:
        raise Exception('expect sim <1')
    Tnet = dict(np.load('%s.npz'%(obj_name)))
    categ = []
    categ_sim = []
    print (len(Tnet))
    for key in Tnet:
        if (len(categ) == 0):
            categ.append([[key,1.0,Tnet[key]]])
        else:
            flag = True
            for ct in categ:
                avg_similarity = cosine_distance(Tnet[ct[0][0]], Tnet[key])
                if (avg_similarity > sim):
                    ct.append([key,avg_similarity,Tnet[key]])
                    flag = False
                    break
            if (flag):
                categ.append([[key,1.0,Tnet[key]]])
            print (avg_similarity)

    # print(json.dumps(categ, cls=NumpyEncoder))
    return categ

#根据分类移动图片文件(需要把result下的图片文件夹到当前目录下)
def mvpic(id,pic_dir,aim_dir):
    filename1 = NAME_STR1 % (id)
    filename1 = os.path.join(pic_dir, filename1)
    if os.path.isfile(filename1):
        shutil.move(filename1, aim_dir)

    filename2 = NAME_STR2 % (id)
    filename2 = os.path.join(pic_dir, filename2)
    if os.path.isfile(filename2):
        shutil.move(filename2, aim_dir)

#根据分类复制图片文件(需要把result下的图片文件夹复制过来)
def cppic(id,pic_dir,aim_dir):
    filename1 = NAME_STR1 % (id)
    filename1 = os.path.join(pic_dir, filename1)

    filename1 = glob.glob(filename1)[0]
    if os.path.isfile(filename1):
        shutil.copy(filename1, aim_dir)

    filename2 = NAME_STR2 % (id)
    filename2 = os.path.join(pic_dir, filename2)

    filename2 = glob.glob(filename2)[0]
    if os.path.isfile(filename2):
        shutil.copy(filename2, aim_dir)

if __name__=='__main__':
    pic_dir = os.path.join(PICS_DIR, OBJ_NAME)

    obj_dir = os.path.join(BASE_DIR, OBJ_NAME)
    if os.path.exists(obj_dir): shutil.rmtree(obj_dir)
    os.mkdir(obj_dir)

    log_fout = open(os.path.join(obj_dir, 'log.txt'), 'w')
    log_fout.write(str(FLAGS) + '\n')

    obj_file = os.path.join(RLG_DIR,OBJ_NAME)
    categ = category_by_avg_similarity(obj_file,FLAGS.sim)

    log_fout.write('category:'+len(categ).__str__()+'\n')
    log_fout.flush()
    print ('category:'+len(categ).__str__())

    ctcnt = 0
    for ct in categ:
        log_fout.write(ctcnt.__str__()+': '+len(ct).__str__() + '\n')
        log_fout.flush()
        print (ctcnt.__str__()+': '+len(ct).__str__())

        aim_dir = os.path.join(obj_dir, ctcnt.__str__())
        if not os.path.exists(aim_dir): os.mkdir(aim_dir)

        #输出json格式的categ
        json_file = "%d.json"%(ctcnt)
        json_file = os.path.join(aim_dir, json_file)
        # print (json_file)
        json.dump(categ[ctcnt], codecs.open(json_file, 'w', encoding='utf-8'), sort_keys=True, indent=4,cls=NumpyEncoder)

        ctcnt += 1

        #根据分类移动图片文件
        for id_s in ct:
            cppic(id_s[0],pic_dir,aim_dir)
            # print (filename)
    # print (PAR_DIR)
