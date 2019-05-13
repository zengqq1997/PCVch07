# PCVch07

python计算机视觉第七章实验
第八次记录
## 图像检索

### 准备工作

- 需下载数据集[first1000](http://www.vis.uky.edu/~stewe/ukbench/)
- 安装pysqlite（pip install pysqlite）
- 安装CherryPy（pip install cherryPy -i https://pypi.tuna.tsinghua.edu.cn/simple）

### 引言

在Web2.0时代，尤其是随着Flickr、Facebook等社交网站的流行，图像、视频、音频、文本等异构数据每天都在以惊人的速度增长。例如， Facebook注册用户超过10亿，每月上传超过10亿的图片；Flickr图片社交网站2015年用户上传图片数目达7.28亿，平均每天用户上传约200万的图片；中国最大的电子商务系统淘宝网的后端系统上保存着286亿多张图片。针对这些包含丰富视觉信息的海量图片，如何在这些浩瀚的图像库中方便、快速、准确地查询并检索到用户所需的或感兴趣的图像，成为多媒体信息检索领域研究的热点。基于内容的图像检索方法充分发挥了计算机长于处理重复任务的优势，将人们从需要耗费大量人力、物力和财力的人工标注中解放出来。经过十来来的发展，基于内容的图像检索技术已广泛应用于搜索引擎、电子商务、医学、纺织业、皮革业等生活的方方面面。

图像检索按描述图像内容方式的不同可以分为两类，一类是**基于文本的图像检索**(TBIR, Text Based Image Retrieval)，另一类是**基于内容的图像检索**(CBIR, Content Based Image Retrieval)。

### 基于内容的图像检索

对图像的内容语义，如图像的颜色、纹理、布局等进行分析和检索的图像检索技术，即基于内容的图像检索（Content-based Image Retrieval，简称CBIR）技术。CBIR属于基于内容检索（Content-based Retrieval，简称CBR）的一种，CBR中还包括对动态视频、音频等其它形式多媒体信息的检索技术。

### BOW模型

Bag-of-words在CV中的应用首先出现在Andrew Zisserman中为解决对视频场景的搜索，其提出了使用Bag-of-words关键点投影的方法来表示图像信息。后续更多的研究者归结此方法为Bag-of-Features，并用于图像分类、目标识别和图像检索。Bag-of-Features模型仿照文本检索领域的Bag-of-Words方法，把每幅图像描述为一个局部区域或关键点(Patches/Key Points)特征的无序集合，这些特征点可以看成一个词。这样，就能够把文本检索及分类的方法用到图像分类及检索中去。

Bag-of-Words模型源于文本分类技术。在信息检索中，它假定对于一个文本，忽略其词序、语法和句法，将其仅仅看作是一个词集合，或者说是词的一个组合。文本中每个词的出现都是独立的，不依赖于其他词是否出现，或者说在任意一个位置选择词汇都不受前面句子的影响而独立选择的。

使用某种聚类算法(如K-means)将特征进行聚类，每个聚类中心被看作是词典中的一个视觉单词，相当于文本检索中的词，视觉词汇由聚类中心对应特征形成的码字来表示。

#### 实验步骤

> 1. 图像预处理，提取图像特征
> 2. 对图像数据库建立图像特征数据库
> 3. 抽取检索图像特征，构建特征向量
> 4. 建立演示程序及web应用

##### 图像预处理，提取图像特征

```python
# -*- coding: utf-8 -*-
import pickle
from PCV.imagesearch import vocabulary
from PCV.tools.imtools import get_imlist
from PCV.localdescriptors import sift

#获取图像列表
imlist = get_imlist('first1000/')
nbr_images = len(imlist)
#获取特征列表
featlist = [imlist[i][:-3]+'sift' for i in range(nbr_images)]

#提取文件夹下图像的sift特征
for i in range(nbr_images):
    sift.process_image(imlist[i], featlist[i])

#生成词汇
voc = vocabulary.Vocabulary('ukbenchtest')
voc.train(featlist, 1000, 10)
#保存词汇
# saving vocabulary
with open('first1000/vocabulary.pkl', 'wb') as f:
    pickle.dump(voc, f)
print ('vocabulary is:', voc.name, voc.nbr_words)
```

**结果**得到了一千张图像的SIFT特征数据并且生成了视觉词典vocabulary.pkl



##### 建立图像特征数据库

```Python
# -*- coding: utf-8 -*
import pickle
from PCV.imagesearch import imagesearch
from PCV.localdescriptors import sift
from sqlite3 import dbapi2 as sqlite
from PCV.tools.imtools import get_imlist

##要记得将PCV放置在对应的路径下
##要记得将PCV放置在对应的路径下

# 获取图像列表
imlist = get_imlist('./data/')##记得改成自己的路径
nbr_images = len(imlist)
#获取特征列表
featlist = [imlist[i][:-3]+'sift' for i in range(nbr_images)]

# load vocabulary
#载入词汇
with open('./data/vocabulary.pkl', 'rb') as f:
    voc = pickle.load(f)
#创建索引
indx = imagesearch.Indexer('testImaAdd.db',voc)
indx.create_tables()
# go through all images, project features on vocabulary and insert
#遍历所有的图像，并将它们的特征投影到词汇上
for i in range(nbr_images)[:1000]:
    locs,descr = sift.read_features_from_file(featlist[i])
    indx.add_to_index(imlist[i],descr)
# commit to database
#提交到数据库
indx.db_commit()

con = sqlite.connect('testImaAdd.db')
print(con.execute('select count (filename) from imlist').fetchone())
print(con.execute('select * from imlist').fetchone())
```

**结果**生成了testImaAdd.db数据库



##### 检索图像特征

```PYTHon
# -*- coding: utf-8 -*
import pickle
from PCV.localdescriptors import sift
from PCV.imagesearch import imagesearch
from PCV.geometry import homography
from PCV.tools.imtools import get_imlist

##要记得将PCV放置在对应的路径下
##要记得将PCV放置在对应的路径下

# load image list and vocabulary
#载入图像列表
imlist = get_imlist('./data/')##要改成自己的地址
nbr_images = len(imlist)

#载入特征列表
featlist = [imlist[i][:-3]+'sift' for i in range(nbr_images)]

# 载入词汇
with open('./data/vocabulary.pkl', 'rb') as f:     ##要改成自己的地址
    voc = pickle.load(f)

src = imagesearch.Searcher('testImaAdd.db', voc)

# index of query image and number of results to return
#查询图像索引和查询返回的图像数
q_ind = 0
nbr_results = 20

# regular query
# 常规查询(按欧式距离对结果排序)
res_reg = [w[1] for w in src.query(imlist[q_ind])[:nbr_results]]
print('top matches (regular):', res_reg)

# load image features for query image
#载入查询图像特征
q_locs,q_descr = sift.read_features_from_file(featlist[q_ind])
fp = homography.make_homog(q_locs[:,:2].T)

# RANSAC model for homography fitting
#用单应性进行拟合建立RANSAC模型
model = homography.RansacModel()
rank = {}

# load image features for result
#载入候选图像的特征
for ndx in res_reg[1:]:
    locs,descr = sift.read_features_from_file(featlist[ndx])  # because 'ndx' is a rowid of the DB that starts at 1
# get matches
    matches = sift.match(q_descr, descr)
    ind = matches.nonzero()[0]
    ind2 = matches[ind]
    tp = homography.make_homog(locs[:,:2].T)
    # compute homography, count inliers. if not enough matches return empty list
    try:
        H,inliers = homography.H_from_ransac(fp[:,ind],tp[:,ind2],model,match_theshold=4)
    except:
        inliers = []
    # store inlier count
    rank[ndx] = len(inliers)

# sort dictionary to get the most inliers first
sorted_rank = sorted(rank.items(), key=lambda t: t[1], reverse=True)
res_geom = [res_reg[0]]+[s[0] for s in sorted_rank]
print('top matches (homography):', res_geom)

# 显示查询结果
imagesearch.plot_results(src,res_reg[:8]) #常规查询
imagesearch.plot_results(src,res_geom[:8]) #重排后的结果
```

**结果**

![img](https://github.com/zengqq1997/PCVch07/blob/master/result3.jpg)

![img](https://github.com/zengqq1997/PCVch07/blob/master/result2.jpg)

![img](https://github.com/zengqq1997/PCVch07/blob/master/result1.jpg)



##### 建立演示程序及web应用

```python
# -*- coding: utf-8 -*
import cherrypy
import pickle
import urllib
import os
from numpy import *
#from PCV.tools.imtools import get_imlist
from PCV.imagesearch import imagesearch
import random

""" This is the image search demo in Section 7.6. """


class SearchDemo:

    def __init__(self):
        # 载入图像列表
        self.path = './data/'
        #self.path = 'D:/python_web/isoutu/first500/'
        self.imlist = [os.path.join(self.path,f) for f in os.listdir(self.path) if f.endswith('.jpg')]
        #self.imlist = get_imlist('./first500/')
        #self.imlist = get_imlist('E:/python/isoutu/first500/')
        self.nbr_images = len(self.imlist)
        print(self.imlist)
        print(self.nbr_images)
        #print(str(len(self.imlist))+"###############")
        #self.ndx = range(self.nbr_images)
        self.ndx = list(range(self.nbr_images))
        print(self.ndx)

        # 载入词汇
        # f = open('first1000/vocabulary.pkl', 'rb')
        with open('./data/vocabulary.pkl','rb') as f:
            self.voc = pickle.load(f)
        #f.close()

        # 显示搜索返回的图像数
        self.maxres = 10

        # header and footer html
        self.header = """
            <!doctype html>
            <head>
            <title>Image search</title>
            </head>
            <body>
            """
        self.footer = """
            </body>
            </html>
            """

    def index(self, query=None):
        self.src = imagesearch.Searcher('testImaAdd.db', self.voc)

        html = self.header
        html += """
            <br />
            Click an image to search. <a href='?query='> Random selection </a> of images.
            <br /><br />
            """
        if query:

            # query the database and get top images
            # 查询数据库，并获取前面的图像
            res = self.src.query(query)[:self.maxres]
            for dist, ndx in res:
                imname = self.src.get_filename(ndx)
                html += "<a href='?query="+imname+"'>"

                html += "<img src='"+imname+"' alt='"+imname+"' width='100' height='100'/>"
                print(imname+"################")
                html += "</a>"
            # show random selection if no query
            # 如果没有查询图像则随机显示一些图像
        else:
            random.shuffle(self.ndx)
            for i in self.ndx[:self.maxres]:
                imname = self.imlist[i]
                html += "<a href='?query="+imname+"'>"

                html += "<img src='"+imname+"' alt='"+imname+"' width='100' height='100'/>"
                print(imname+"################")
                html += "</a>"

        html += self.footer
        return html

    index.exposed = True

# conf_path = os.path.dirname(os.path.abspath(__file__))
#conf_path = os.path.join(conf_path, "service.conf")
#cherrypy.config.update(conf_path) #cherrypy.quickstart(SearchDemo())

cherrypy.quickstart(SearchDemo(), '/', config=os.path.join(os.path.dirname(__file__), 'service.conf'))
```

**结果**

![img](https://github.com/zengqq1997/PCVch07/blob/master/result4.jpg)

复制红色箭头所指的地址，从浏览器中打开

![img](https://github.com/zengqq1997/PCVch07/blob/master/result5.jpg)

![img](https://github.com/zengqq1997/PCVch07/blob/master/result6.jpg)



#### 遇到的问题

1. 在安装cherrypy的时候，用pip install CherryPy命令行老是下载失败，直接下载包也是失败，不知道到底是什么原因，后来用国内的镜像下载就可以成功了pip install cherryPy -i ![img]pip install cherryPy -i https://pypi.tuna.tsinghua.edu.cn/simple

​      ![img](https://github.com/zengqq1997/PCVch07/blob/master/error.jpg)

2. 在运行建立数据库部分代码时，因为PCV文件夹中的imagesearch夹下的imagesearch.py中import 没有及时更新导致报错，将其更改为import sqlite3即可。

   ![img](https://github.com/zengqq1997/PCVch07/blob/master/error1.jpg)

   
