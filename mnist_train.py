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

# #### 2. mnist_train.py

# In[6]:


import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference


# 1. 配置神经网络结构相关参数。

# In[7]:


BATCH_SIZE = 100 
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 20000
MOVING_AVERAGE_DECAY = 0.99 
# 模型保存路径和文件名
MODEL_SAVE_PATH = "C:/Users/74575/Desktop/pythonfile/MNIST_model/"
MODEL_NAME = "mnist_model.ckpt"


# 2. 定义训练过程，支持程序关闭后从checkpoint恢复训练。

# In[8]:


def train(mnist):
    # 定义checkpoint保存点
    ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
    # 定义输入输出placeholder。
    x = tf.placeholder(
        tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(
        tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # 直接使用mnist_inference.py中定义的前向传播过程
    y = mnist_inference.inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)
    
    # 定义损失函数、学习率、滑动平均操作以及训练过程。
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(
        tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE, 
        LEARNING_RATE_DECAY,
        staircase=True)
    train_step = tf.train.GradientDescentOptimizer(learning_rate)                   .minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')
        
    # 初始化TensorFlow持久化类。
    saver = tf.train.Saver()  
    with tf.Session() as sess:
        saved_step = 0
        if ckpt and ckpt.model_checkpoint_path:
            print("checkpoint存在，直接恢复变量")
            saver.restore(sess, ckpt.model_checkpoint_path)
            # 恢复global_step
            saved_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            sess.run(global_step.assign(saved_step))
        else:
            print("checkpoint不存在，进行变量初始化")
            tf.global_variables_initializer().run()

        for i in range(saved_step, TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
                last_step = step - 1000
                if last_step > 0:
                    try:
                        os.remove(MODEL_SAVE_PATH+MODEL_NAME+"-"+str(last_step)+".index")
                        os.remove(MODEL_SAVE_PATH+MODEL_NAME+"-"+str(last_step)+".data-00000-of-00001")
                        os.remove(MODEL_SAVE_PATH+MODEL_NAME+"-"+str(last_step)+".meta")
                    except:
                        print("删除数据异常")
                    else:
                        print("成功删除：", MODEL_SAVE_PATH+MODEL_NAME+"-"+str(last_step)+".*")


# 3. 主程序入口

# In[9]:


tf.reset_default_graph() # 这玩意儿很重要！
def main(argv=None):
    mnist = input_data.read_data_sets("../../datasets/MNIST_data", one_hot=True)
    train(mnist)

if __name__ == '__main__':
    main()

