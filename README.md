# 图像处理 100 问！！

> 日本语本当苦手，翻译出错还请在 issue 指正。代码算法方面的问题请往原[ repo ](https://github.com/yoyoyo-yo/Gasyori100knock)提。现阶段我并没有做这些题目（捂脸……），只是翻译而已，因此算法细节可能没有翻译到位。不太好翻译的地方我也会在一定程度上意译~~自行发挥~~，请各位谅解。后续在写代码的途中会对翻译有所更正。
>
> 我会尽量附上英文术语，有翻译不清楚的地方还请参照原文、英语及代码。
>
> 感谢！
>
> ——gzr

英文版本在[这里]( https://github.com/KuKuXia/Image_Processing_100_Questions)，谢谢[KuKuXia](https://github.com/KuKuXia)桑为我做英文翻译。

为图像处理初学者设计的 100 个问题完成了啊啊啊啊啊(´；ω；｀)

和蝾螈一起学习基本的图像处理知识，理解图像处理算法吧！解答这里的提出的问题请不要调用`OpenCV`的`API`，**自己动手实践吧**！虽然包含有答案，但不到最后请不要参考。一边思考，一边完成这些问题吧！

- **问题不是按照难易程度排序的。虽然我尽可能地提出现在流行的问题，但在想不出新问题的情况下也会提出一些没怎么听说过的问题（括弧笑）。**

- **这里的内容参考了各式各样的文献，因此也许会有不对的地方，请注意！**如果发现了错误还请 pull requests ！！

- 【注意】使用这个页面造成的任何事端，本人不负任何责任。

  > 俺也一样。使用这个页面造成的任何事端，本人不负任何责任。
  >
  > ——gzr

请根据自己的喜好，选择 Python 或者 C++ 来进行尝试吧。

> 深度学习无限问请点击[这里](https://github.com/yoyoyo-yo/DeepLearningMugenKnock)。

## Recent
- 2019.3.13 Q95-100 Neural Networkを修正
- 2019.3.8 Questions_01_10 にC++の解答を追加！
- 2019.3.7 TutorialにC++用を追加　そろそろC++用の答案もつくろっかなーと
- 2019.3.5 各Questionの答案をanswersディレクトリに収納
- 2019.3.3 Q.18-22. 一部修正
- 2019.2.26 Q.10. メディアンフィルタの解答を一部修正
- 2019.2.25 Q.9. ガウシアンフィルタの解答を一部修正
- 2019.2.23 Q.6. 減色処理のREADMEを修正
- 2019.1.29 HSVを修正

## 首先

打开终端，输入以下指令。使用这个命令，你可以将整个目录完整地克隆到你的计算机上。

```bash
$ git clone https://github.com/yoyoyo-yo/Gasyori100knock.git
```

然后，选择你喜欢的 Python 或者 C++，阅读下一部分——Tutorial！

## [Tutorial](https://github.com/yoyoyo-yo/Gasyori100knock/tree/master/Tutorial)

|       |      内容      |                                                                     Python                                                                      |                                                                           C++                                                                            |
| :---: | :------------: | :---------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   1   |      安装      |                                     [✓](https://github.com/yoyoyo-yo/Gasyori100knock/tree/master/Tutorial)                                      |                            [✓](https://github.com/yoyoyo-yo/Gasyori100knock/blob/master/Tutorial/README_opencv_c_install.md)                             |
|   2   | 读取、显示图像 | [✓](https://github.com/yoyoyo-yo/Gasyori100knock/tree/master/Tutorial#%E7%94%BB%E5%83%8F%E8%AA%AD%E3%81%BF%E8%BE%BC%E3%81%BF%E8%A1%A8%E7%A4%BA) | [✓](https://github.com/yoyoyo-yo/Gasyori100knock/blob/master/Tutorial/README_opencv_c_install.md#%E7%94%BB%E5%83%8F%E8%AA%AD%E3%81%BF%E8%BE%BC%E3%81%BF) |
|   3   |    操作像素    |          [✓](https://github.com/yoyoyo-yo/Gasyori100knock/tree/master/Tutorial#%E7%94%BB%E7%B4%A0%E3%82%92%E3%81%84%E3%81%98%E3%82%8B)          | [✓](https://github.com/yoyoyo-yo/Gasyori100knock/blob/master/Tutorial/README_opencv_c_install.md#%E7%94%BB%E7%B4%A0%E3%82%92%E3%81%84%E3%81%98%E3%82%8B) |
|   4   |    拷贝图像    |          [✓](https://github.com/yoyoyo-yo/Gasyori100knock/tree/master/Tutorial#%E7%94%BB%E5%83%8F%E3%81%AE%E3%82%B3%E3%83%94%E3%83%BC)          | [✓](https://github.com/yoyoyo-yo/Gasyori100knock/blob/master/Tutorial/README_opencv_c_install.md#%E7%94%BB%E5%83%8F%E3%81%AE%E3%82%B3%E3%83%94%E3%83%BC) |
|   5   |    保存图像    |              [✓](https://github.com/yoyoyo-yo/Gasyori100knock/tree/master/Tutorial#%E7%94%BB%E5%83%8F%E3%81%AE%E4%BF%9D%E5%AD%98)               |     [✓](https://github.com/yoyoyo-yo/Gasyori100knock/blob/master/Tutorial/README_opencv_c_install.md#%E7%94%BB%E5%83%8F%E3%81%AE%E4%BF%9D%E5%AD%98)      |
|   6   |    练习问题    |                            [✓](https://github.com/yoyoyo-yo/Gasyori100knock/tree/master/Tutorial#%E7%B7%B4%E7%BF%92)                            |          [✓](https://github.com/yoyoyo-yo/Gasyori100knock/blob/master/Tutorial/README_opencv_c_install.md#%E7%B7%B4%E7%BF%92%E5%95%8F%E9%A1%8C)          |

请在这之后解答提出的问题。问题内容分别包含在各个文件夹中。请使用示例图片`assets/imori.jpg`。在各个文件夹中的`README.md`里有问题和解答。运行答案，请使用以下指令（自行替换文件夹和文件名）：

```python
python answers/answer_@@.py
```

## 问题

详细的问题请参见各页面下的`README`文件（各个页面下滑就可以看见）。
- 为了简化答案，所以没有编写`main()`函数。
- 虽然我们的答案以`numpy`为基础，但是还请你自己查找`numpy`的基本使用方法。

### [問題1 - 10](https://github.com/yoyoyo-yo/Gasyori100knock/tree/master/Question_01_10)

| 序号  |            问题             |                                              Python                                               |                                                  C++                                                   |
| :---: | :-------------------------: | :-----------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------: |
|   1   |          通道替换           | [✓](https://github.com/yoyoyo-yo/Gasyori100knock/tree/master/Question_01_10/answers/answer_1.py)  | [✓](https://github.com/yoyoyo-yo/Gasyori100knock/tree/master/Question_01_10/answers_cpp/answer_1.cpp)  |
|   2   |     灰度化（Grayscale）     | [✓](https://github.com/yoyoyo-yo/Gasyori100knock/tree/master/Question_01_10/answers/answer_2.py)  | [✓](https://github.com/yoyoyo-yo/Gasyori100knock/tree/master/Question_01_10/answers_cpp/answer_2.cpp)  |
|   3   |   二值化（Thresholding）    | [✓](https://github.com/yoyoyo-yo/Gasyori100knock/tree/master/Question_01_10/answers/answer_3.py)  | [✓](https://github.com/yoyoyo-yo/Gasyori100knock/tree/master/Question_01_10/answers_cpp/answer_3.cpp)  |
|   4   |          大津算法           | [✓](https://github.com/yoyoyo-yo/Gasyori100knock/tree/master/Question_01_10/answers/answer_4.py)  | [✓](https://github.com/yoyoyo-yo/Gasyori100knock/tree/master/Question_01_10/answers_cpp/answer_4.cpp)  |
|   5   |          HSV 变换           | [✓](https://github.com/yoyoyo-yo/Gasyori100knock/tree/master/Question_01_10/answers/answer_5.py)  | [✓](https://github.com/yoyoyo-yo/Gasyori100knock/tree/master/Question_01_10/answers_cpp/answer_5.cpp)  |
|   6   |          减色处理           | [✓](https://github.com/yoyoyo-yo/Gasyori100knock/tree/master/Question_01_10/answers/answer_6.py)  | [✓](https://github.com/yoyoyo-yo/Gasyori100knock/tree/master/Question_01_10/answers_cpp/answer_6.cpp)  |
|   7   | 平均池化（Average Pooling） | [✓](https://github.com/yoyoyo-yo/Gasyori100knock/tree/master/Question_01_10/answers/answer_7.py)  | [✓](https://github.com/yoyoyo-yo/Gasyori100knock/tree/master/Question_01_10/answers_cpp/answer_7.cpp)  |
|   8   |   最大池化（Max Pooling）   | [✓](https://github.com/yoyoyo-yo/Gasyori100knock/tree/master/Question_01_10/answers/answer_8.py)  | [✓](https://github.com/yoyoyo-yo/Gasyori100knock/tree/master/Question_01_10/answers_cpp/answer_8.cpp)  |
|   9   | 高斯滤波（Gaussian Filter） | [✓](https://github.com/yoyoyo-yo/Gasyori100knock/tree/master/Question_01_10/answers/answer_9.py)  | [✓](https://github.com/yoyoyo-yo/Gasyori100knock/tree/master/Question_01_10/answers_cpp/answer_9.cpp)  |
|  10   |  中值滤波（Median filter）  | [✓](https://github.com/yoyoyo-yo/Gasyori100knock/tree/master/Question_01_10/answers/answer_10.py) | [✓](https://github.com/yoyoyo-yo/Gasyori100knock/tree/master/Question_01_10/answers_cpp/answer_10.cpp) |

## [问题11 - 20](https://github.com/yoyoyo-yo/Gasyori100knock/tree/master/Question1120)

| 序号 |      内容      |
| :--: | :------------: |
|  11  |    均值滤波    |
|  12  | Motion Filter  |
|  13  |  MAX-MIN 滤波  |
|  14  |    微分滤波    |
|  15  |   Sobel 滤波   |
|  16  |  Prewitt 滤波  |
|  17  | Laplacian 滤波 |
|  18  |  Emboss 滤波   |
|  19  |    LoG 滤波    |
|  20  |   直方图表示   |

## 问题21-30

| 序号 |                     内容                     |
| :--: | :------------------------------------------: |
|  21  |   直方图归一化（Histogram Normalization）    |
|  22  |                  直方图操作                  |
|  23  |    直方图均衡化（Histogram Equalization）    |
|  24  |         伽玛校正（Gamma Correction）         |
|  25  | 最邻近插值（Nearest-neighbor Interpolation） |
|  26  |     双线性插值（Bilinear Interpolation）     |
|  27  |     双三次插值（Bicubic Interpolation）      |
|  28  | 仿射变换（Afine Transformations）——平行移动  |
|  29  | 仿射变换（Afine Transformations）——放大缩小  |
|  30  |   仿射变换（Afine Transformations）——旋转    |

## 问题31-40

| 序号 |                             内容                             |
| :--: | :----------------------------------------------------------: |
|  31  |           仿射变换（Afine Transformations）——倾斜            |
|  32  |               傅立叶变换（Fourier Transform）                |
|  33  |                     傅立叶变换——低通滤波                     |
|  34  |                     傅立叶变换——高通滤波                     |
|  35  |                     傅立叶变换——带通滤波                     |
|  36  | JPEG 压缩——第一步：离散余弦变换（Discrete Cosine Transformation） |
|  37  |           峰值信噪比（Peak Signal to Noise Ratio）           |
|  38  |             JPEG 压缩——第二步：离散余弦变换+量化             |
|  39  |              JPEG 压缩——第三步：YCbCr 色彩空间               |
|  40  |              JPEG 压缩——第四步：YCbCr+DCT+量化               |

## 问题41-50

| 序号 |                           内容                            |
| :--: | :-------------------------------------------------------: |
|  41  |             `Canny`边缘检测：第一步——边缘强度             |
|  42  |             `Canny`边缘检测：第二步——边缘细化             |
|  43  |             `Canny`边缘检测：第三步——滞后阈值             |
|  44  |  霍夫变换（Hough Transform）／直线检测——第一步：霍夫变换  |
|  45  |    霍夫变换（Hough Transform）／直线检测——第二步：NMS     |
|  46  | 霍夫变换（Hough Transform）／直线检测——第三步：霍夫逆变换 |
|  47  |                形态学处理：膨胀（Dilate）                 |
|  48  |                 形态学处理：腐蚀（Erode）                 |
|  49  |                开运算（Opening Operation）                |
|  50  |                闭运算（Closing Operation）                |

## 问题51-60

| 序号 |                             内容                             |
| :--: | :----------------------------------------------------------: |
|  51  |              形态学梯度（Morphology Gradient）               |
|  52  |                       顶帽（Top Hat）                        |
|  53  |                      黑帽（Black Hat）                       |
|  54  | 使用误差平方和算法（Sum of Squared Difference）进行模式匹配（Template Matching） |
|  55  |  使用绝对值差和（Sum of Absolute Differences）进行模式匹配   |
|  56  | 使用归一化交叉相关（Normalization Cross Correlation）进行模式匹配 |
|  57  | 使用零均值归一化交叉相关（Zero-mean Normalization Cross Correlation）进行模式匹配 |
|  58  |                       4-邻接连通域标记                       |
|  59  |                       8-邻接连通域标记                       |
|  60  |                  透明混合（Alpha Blending）                  |

## 问题61-70

| 序号 |                      内容                       |
| :--: | :---------------------------------------------: |
|  61  |                 4-邻接的连接数                  |
|  62  |                 8-邻接的连接数                  |
|  63  |                    细化处理                     |
|  64  |                Hilditch 细化算法                |
|  65  |               Zhang-Suen 细化算法               |
|  66  | 方向梯度直方图（HOG）第一步：梯度幅值・梯度方向 |
|  67  |     方向梯度直方图（HOG）第二步：梯度直方图     |
|  68  |    方向梯度直方图（HOG）第三步：直方图归一化    |
|  69  |    方向梯度直方图（HOG）第四步：可视化特征量    |
|  70  |           色彩追踪（Color Tracking）            |

## 问题71-80

| 序号 |                     内容                      |
| :--: | :-------------------------------------------: |
|  71  |                掩膜（Masking）                |
|  72  | 掩膜（色彩追踪（Color Tracking）+形态学处理） |
|  73  |                  缩小和放大                   |
|  74  |          使用差分金字塔提取高频成分           |
|  75  |        高斯金字塔（Gaussian Pyramid）         |
|  76  |            显著图（Saliency Map）             |
|  77  |         Gabor 滤波器（Gabor Filter）          |
|  78  |               旋转 Gabor 滤波器               |
|  79  |         使用 Gabor 滤波器进行边缘检测         |
|  80  |         使用 Gabor 滤波器进行特征提取         |

## 问题81-90

| 序号 |                           内容                            |
| :--: | :-------------------------------------------------------: |
|  81  |                     Hessian 角点检测                      |
|  82  |          Harris 角点检测第一步：Sobel + Gausian           |
|  83  |              Harris 角点检测第二步：角点检测              |
|  84  |             简单图像识别第一步：减色化+直方图             |
|  85  |               简单图像识别第二步：判别类别                |
|  86  |                 简单图像识别第三步：评估                  |
|  87  |                 简单图像识别第四步：k-NN                  |
|  88  |   k-平均聚类算法（k -means Clustering）第一步：生成质心   |
|  89  |     k-平均聚类算法（k -means Clustering）第二步：聚类     |
|  90  | k-平均聚类算法（k -means Clustering）第三步：调整初期类别 |

## 问题91-100

| 序号 |                             内容                             |
| :--: | :----------------------------------------------------------: |
|  91  |    利用 k-平均聚类算法进行减色处理第一步：按颜色距离分类     |
|  92  |       利用 k-平均聚类算法进行减色处理第二步：减色处理        |
|  93  |            准备机器学习的训练数据第一步：计算 IoU            |
|  94  |  准备机器学习的训练数据第一步：随机裁剪（Random Cropping）   |
|  95  | 神经网络（Neural Network）第一步：深度学习（Deep Learning）  |
|  96  |            神经网络（Neural Network）第二步：训练            |
|  97  |     简单物体检测第一步----滑动窗口（Sliding Window）+HOG     |
|  98  |     简单物体检测第二步----滑动窗口（Sliding Window）+ NN     |
|  99  | 简单物体检测第三步----非极大值抑制（Non-Maximum Suppression） |
| 100  |  简单物体检测第三步----评估 Precision, Recall, F-score, mAP  |

## TODO

1. 问题47、48待翻译
2. 问题81待翻译
3. 问题100待翻译
4. 链接修复

## Citation

```bash
@article{yoyoyo-yoGasyori100knock,
    Author = {yoyoyo-yo},
    Title = {Gasyori100knock},
    Journal = {https://github.com/yoyoyo-yo/Gasyori100knock},
    Year = {2019}
}
```

