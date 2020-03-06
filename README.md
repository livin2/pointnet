# PointNet分析验证实验

比较不同数据下input transform所用的3x3矩阵之间的相似度，将输出按照相似的旋转矩阵分类。

## 编码清单

- analysis/ :分析用的工具脚本([见该目录下的README](analysis/README.md))
  - diff_species 不同类分析
  - same_species 同类分析
- pointnet/：对原有的pointnet的修改
- result/: 上面修改后pointnet运行后输出的结果:
  - final/: 按物体类别分类的经过Tnet旋转后与旋转前的物体三视图，文件名格式如下
    - 编号\_旋转前后\_label\_数据原类别\_pred\_模型识别类别.jpg
    - 104_beforeR_label_flower_pot_pred_plant.jpg
    - 104_TnetR_label_flower_pot_pred_plant.jpg
  - rot_log/: 若干个‘类别’.npz与'类别_evaluate.txt' 内容都为其类别每个物体旋转所使用的3x3矩阵 txt中每个矩阵的标签对应final中的图片文件名
  - log_evaluate.txt 执行evaluate.py输出的日志

