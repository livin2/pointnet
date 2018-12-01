# -*- coding:utf-8 -*- 
import numpy as np
import os
import shutil
import json
import argparse
import codecs
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--obj', default='airplane')
parser.add_argument('--sim',type=float, default=0.9)
FLAGS = parser.parse_args()

OBJ_NAME = FLAGS.obj
NAME_STR1 = '%s_beforeR_label_'+OBJ_NAME+'_pred_'+OBJ_NAME+'.jpg'
NAME_STR2 = '%s_TnetR_label_'+OBJ_NAME+'_pred_'+OBJ_NAME+'.jpg'

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

#根据分类移动图片文件(需要把result下的图片文件夹复制过来)
def mvpic(id):
    filename1 = NAME_STR1 % (id)
    filename1 = os.path.join(pic_dir, filename1)
    if os.path.isfile(filename1):
        shutil.move(filename1, aim_dir)

    filename2 = NAME_STR2 % (id)
    filename2 = os.path.join(pic_dir, filename2)
    if os.path.isfile(filename2):
        shutil.move(filename2, aim_dir)

if __name__=='__main__':
    categ = category_by_avg_similarity(OBJ_NAME,FLAGS.sim)

    print (len(categ))

    ctcnt = 0
    pic_dir = os.path.join(BASE_DIR, OBJ_NAME)
    for ct in categ:
        aim_dir = os.path.join(pic_dir, ctcnt.__str__())
        if not os.path.exists(aim_dir): os.mkdir(aim_dir)

        #输出json格式的categ
        json_file = "file.json"
        json_file = os.path.join(aim_dir, json_file)
        # print (json_file)
        json.dump(categ[ctcnt], codecs.open(json_file, 'w', encoding='utf-8'), sort_keys=True, indent=4,cls=NumpyEncoder)

        ctcnt += 1

        #根据分类移动图片文件
        for id_s in ct:
            mvpic(id_s[0])
            # print (filename)
