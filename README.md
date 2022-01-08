# TensorFlow 变更
#### model.predict_classes 在 TensorFlow v2.5 已被淘汰, 应改为
```
np.argmax(model.predict(x_test_norm), axis=-1)
```

# 深度学习 最佳入门迈向AI专题实战

<img src="封面1.jpeg" alt="drawing" width="400"/>

## 第一篇 深度学习导论
### 1	深度学习(Deep Learning)导论
##### 1.1	人工智能的三波浪潮
##### 1.2	AI学习地图
##### 1.3	机器学习应用领域
##### 1.4	机器学习开发流程
##### 1.5	开发环境安装
### 2	神经网络(Neural Network)原理
##### 2.1	必备的数学与统计知识
##### 2.2	线性代数(Linear Algebra)
##### 2.3	微积分(Calculus)
##### 2.4	机率(Probability)与统计(Statistics)
##### 2.5	线性规划(Linear Programming)
##### 2.6	最小平方法(OLS) vs. 最大概似法(MLE)
##### 2.7	神经网络(Neural Network)求解
## 第二篇 TensorFlow基础篇
### 3	TensorFlow 架构与主要功能
##### 3.1	常用的深度学习套件
##### 3.2	TensorFlow 架构
##### 3.3	张量(Tensor)运算
##### 3.4	自动微分(Automatic Differentiation)
##### 3.5	神经层(Neural Network Layer)
### 4	神经网络的理解与实作
##### 4.1	撰写第一支神经网络程序
##### 4.2	Keras模型种类
##### 4.3	神经层(Layer)
##### 4.4	激励函数(Activation Functions)
##### 4.5	损失函数(Loss Functions) 
##### 4.6	优化器(Optimizers)
##### 4.7	效能衡量指标(Metrics)
##### 4.8	超参数调校(Hyperparameter Tuning) 
### 5	TensorFlow其他常用指令
##### 5.1	特征转换(One-hot encoding etc.)
##### 5.2	模型存盘与加载(Save and Load)
##### 5.3	模型汇总与结构图(Summary and Plotting)
##### 5.4	回调函数(Callbacks)
##### 5.5	工作记录与可视化(TensorBoard) 
##### 5.6	模型布署(Deploy) 与 TensorFlow Serving
##### 5.7	TensorFlow Dataset
### 6	卷积神经网络(Convolutional Neural Network, CNN)
##### 6.1	卷积神经网络简介
##### 6.2	卷积(Convolution) 
##### 6.3	各式卷积
##### 6.4	池化层(Pooling Layer)
##### 6.5	CNN模型实作
##### 6.6	影像数据增补(Data Augmentation)
##### 6.7	可解释的AI(eXplainable AI, XAI)
### 7	预先训练的模型(Pre-trained Model)
##### 7.1	预先训练模型的简介
##### 7.2	采用完整的模型
##### 7.3	采用部分的模型
##### 7.4	转移学习(Transfer Learning)
##### 7.5	Batch Normalization层
## 第三篇 进阶的影像应用
### 8	对象侦测(Object Detection)
##### 8.1	图像辨识模型的发展
##### 8.2	滑动窗口(Sliding Window)
##### 8.3	方向梯度直方图(HOG）
##### 8.4	R-CNN改良
##### 8.5	YOLO算法简介
##### 8.6	YOLO环境建置
##### 8.7	以TensorFlow使用YOLO模型
##### 8.8	YOLO模型训练
##### 8.9	SSD算法
##### 8.10	TensorFlow Object Detection API
##### 8.11	总结
### 9	进阶的影像应用
##### 9.1	语义分割(Semantic Segmentation)介绍
##### 9.2	自动编码器(AutoEncoder)
##### 9.3	语义分割(Semantic segmentation)实作
##### 9.4	实例分割(Instance Segmentation)
##### 9.5	风格转换(Style Transfer) --人人都可以是毕加索
##### 9.6	脸部辨识(Facial Recognition)
##### 9.7	光学文字辨识(OCR)
##### 9.8	车牌辨识(ANPR)
##### 9.9	卷积神经网络的缺点
### 10	生成对抗网络 (Generative Adversarial Network, GAN)
##### 10.1	生成对抗网络介绍
##### 10.2	生成对抗网络种类
##### 10.3	DCGAN
##### 10.4	Progressive GAN
##### 10.5	Conditional GAN
##### 10.6	Pix2Pix
##### 10.7	CycleGAN
##### 10.8	GAN挑战
##### 10.9	深度伪造(Deepfake)
## 第四篇 自然语言处理
### 11	自然语言处理的介绍
##### 11.1	词袋(BOW)与TF-IDF
##### 11.2	词汇前置处理
##### 11.3	词向量(Word2Vec)
##### 11.4	GloVe模型
##### 11.5	中文处理
##### 11.6	spaCy套件
### 12	第 12 章 自然语言处理的算法
##### 12.1	循环神经网络(RNN)
##### 12.2	长短期记忆网络(LSTM)
##### 12.3	LSTM重要参数与多层LSTM
##### 12.4	Gate Recurrent Unit (GRU)
##### 12.5	股价预测
##### 12.6	注意力机制(Attention Mechanism)
##### 12.7	Transformer架构
##### 12.8	BERT
##### 12.9	Transformers套件
##### 12.10	总结
### 13	聊天机器人(ChatBot)
##### 13.1	ChatBot类别
##### 13.2	ChatBot设计
##### 13.3	ChatBot实作
##### 13.4	ChatBot工具套件
##### 13.5	Dialogflow实作
##### 13.6	结语
### 14	语音相关应用
##### 14.1	语音基本认识
##### 14.2	语音前置处理
##### 14.3	语音相关的深度学习应用
##### 14.4	自动语音识别
##### 14.5	自动语音识别实作
##### 14.6	结语
## 第五篇 强化学习 (Reinforcement learning)
### 15	强化学习 (Reinforcement learning)
##### 15.1	强化学习的基础
##### 15.2	强化学习模型
##### 15.3	简单的强化学习架构
##### 15.4	Gym套件
##### 15.5	Gym扩充功能
##### 15.6	动态规划(Dynamic Programming)
##### 15.7	值循环(Value Iteration)
##### 15.8	蒙地卡罗(Monte Carlo)
##### 15.9	时序差分(Temporal Difference)
##### 15.10	其他算法
##### 15.11	井字游戏
##### 15.12	木棒台车(CartPole)
##### 15.13	结论

# 范例程序安装说明
### 程序相关的测试数据过大，请至[这里](https://drive.google.com/file/d/1ysZGVFZT2v21lazVo5exxs1aHG89LQRq/view?usp=sharing)下载
### 请解压至 src 目录下

