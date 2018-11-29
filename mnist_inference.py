#!/usr/bin/env python
# coding: utf-8

# ## TensorFlow MNIST问题实践优化

# 选择环境：Anaconda Python 3.5.2  
# 安装Tensorflow：Python 3.5环境下运行pip install --upgrade --ignore-installed tensorflow  
# 参考书籍：《TensorFlow实战Google深度学习框架（第2版）》  
# ipynb格式：点击阅读原文github

# ### 5.5 TensorFlow最佳实践样例程序

# 解决MNIST问题的重构代码拆成3个程序：  
# mnist_inference.py：定义前向传播过程以及神经网络中的参数  
# mnist_train.py：定义神经网络的训练过程  
# mnist_eval.py：定义测试过程

# #### 1. mnist_inference.py

# In[1]:


import tensorflow as tf


# 1. 定义神经网络结构相关的参数。

# In[2]:


INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500


# 2. 通过tf.get_variable函数来获取变量。

# In[4]:


def get_weight_variable(shape, regularizer):
    weights = tf.get_variable(
        "weights", shape, 
        initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None: 
        tf.add_to_collection('losses', regularizer(weights))
    return weights


# 3. 定义神经网络的前向传播过程。

# In[5]:


def inference(input_tensor, regularizer):
    # 第一层神经网络及前向传播
    with tf.variable_scope('layer1'):
        weights = get_weight_variable(
            [INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable(
            "biases", [LAYER1_NODE], 
            initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
        
    # 第二层神经网络及前向传播
    with tf.variable_scope('layer2'):
        weights = get_weight_variable(
            [LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable(
            "biases", [OUTPUT_NODE], 
            initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases
    
    #返回最后前向传播的结果
    return layer2


# 无论是训练时还是测试时，都可以直接调用inference这个函数，而不用关心具体的神经网络结构。
