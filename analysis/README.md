## same_species

### analysis.py

analysis.py 依赖2个文件(从result文件夹中获得):

- ​	'objname'.npz --包含用于旋转对应图像的3X3矩阵数据
- ​	'objname'文件夹 --包含所有旋转前后的数据

analysis.py 接收三个参数:

- ​	--obj, default='airplane'	--指定物体名字
- ​        --cppic  --指定是否复制图片并分类
- ​	--sim,type=float, default=0.9	--分类3x3矩阵时会计算各向量余弦相似度的平均值，低于sim的分为两个类别

analysis.py执行过后'objname'文件夹下的图片会被复制到本目录对应文件夹下分类 并生成一个json文件

json文件格式如下

```json
[
        数据编号:"125", 
        平均余弦相似度:0.99165940284729, 
    	TNet变换用的3X3矩阵:
        [
            [
                7.8553900718688965, 
                -2.296715497970581, 
                1.6773855686187744
            ], 
            [
                0.1361125409603119, 
                12.630939483642578, 
                0.509243905544281
            ], 
            [
                -2.3731377124786377, 
                0.8184998631477356, 
                6.235332012176514
            ]
        ]
    ]
```

### analysis_V2.py

analysis_V2.py 依赖：

- /result/rot_log/*.npz 
- /analysis/diff_species/analysisDiff.py
- analysis/shape_names.txt

analysis_V2.py 接收一个参数:
​	--sim, default=0.99

analysis_V2.py会比较每类物体npz内每个3x3矩阵与除自己外所有同类物体矩阵的余弦相似度 输出到analysisV2out文件夹下 

以下为analysisV2out/log.txt的单条记录举例说明


```json
obj1 num:100 //1类物体数据数量
obj2 num:100 //2类物体数据数量
similar 3X3 NUM:3651 //有几组两两余弦相似度大于sim(此处为0.99)的数据
3651%9900-airplane-airplane-0.990000.json //json命名规则
//json命名意义如下
//3651 上面的NUM
//比较数，同类物体比较时不会比较自身所以总比较数是100x100-100=9900
//输入的1类物体的类别
//输入的2类物体的类别
//sim值
0.368787878788 //上面 相似组数/总比较组数
```

json单条记录举例说明

```json
    [{
            "0sim": 0.9933300018310547'此处是下面两数组的相似度', 
            "1tent": [*'此处是一个3x3数组'], 
            "2tent": [*'此处是一个3x3数组']
    }]
```



## diff_species

### analysisDiff.py

analysisDiff.py依赖：

- /result/rot_log/*.npz 

analysisDiff.py参数：

- --obj1, default='flower_pot'
- --obj2, default='plant'
- --sim,  default=0.99

analysisDiff.py会比较obj1类物体中每个物体与obj2类中每个物体的矩阵相似度，并将相似度大于sim(default=0.99)的矩阵两两输出，具体输出一个json与log.txt在当前目录下，输出格式见上analysis_V2

### analysisAll.py

analysisAll.py 依赖：

- /result/rot_log/*.npz 
- /analysis/diff_species/analysisDiff.py
- analysis/shape_names.txt

analysisAll.py 接收一个参数:
​	--sim, default=0.99

analysisAll计算每个类别的每个物体与其他类别的每个物体的矩阵相似度，输出相似度大于sim(default=0.99)的组别，具体输出若干个json与log.txt在analysisAll目录下，输出格式见上analysis_V2

## other

calshape.py --计算/PointNetI/result/rot_log/下每类物体npz文件的数据大小 输出为calShape.txt

unit_cmp.py --比较两个3x3 矩阵的余弦相似度