#encoding:utf-8
import tensorflow as tf
import numpy as np
import sklearn.preprocessing as prep
from tensorflow.examples.tutorials.mnist import input_data

#标准均匀分布
def xavier_init(fan_in,fan_out,constant=1):
    low=-constant*np.sqrt(6.0/(fan_in+fan_out))
    hight=constant*np.sqrt(6.0/(fan_in+fan_out))
    return tf.random_uniform((fan_in,fan_out),maxval=hight,
minval=low,dtype=tf.float32)



#########class init函数中定义属性，成员函数的第一个值为self
class AdditiveGaussianNoiseAutoencoder(object):
    def __init__(self,n_input,n_hidden,transfer_function=tf.nn.softplus,optimizer=tf.train.AdamOptimizer(),scale=0.1):
 #网络参数
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function  # 激活函数
        self.training_scale = scale  # 噪声水平
        self.weights = dict()

# 网络结构
        with tf.name_scope('Rawinput'):
            self.x=tf.placeholder(tf.float32,[None,self.n_input])

        with tf.name_scope('noiseAdder'):
            self.scale = tf.placeholder(dtype=tf.float32)
            self.noise_x = self.x + self.scale * tf.random_normal((n_input,))
        with tf.name_scope('Encoder'):
            self.weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden), name='weight1')  # <---
            self.weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32), name='bias1')
            self.hidden = self.transfer(
                tf.add(tf.matmul(self.noise_x, self.weights['w1']), self.weights['b1']))
        with tf.name_scope('Reconstruction'):
            self.weights['w2'] = tf.Variable(xavier_init(self.n_hidden, self.n_input), name='weight2')  # <---
            self.weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32), name='bias2')
            self.reconstruction = tf.add(
                tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])

#训练过程
        with tf.name_scope('Loss'):
            self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2))

        with tf.name_scope('train'):
            self.optimizer = optimizer.minimize(self.cost)
            #全局变量初始化
            init=tf.global_variables_initializer()
            self.sess=tf.Session()
            self.sess.run(init)
            print('begin to run session to initializer the variable')

    def partial_fit(self,X):
        #训练并计算cost
        #数据数据 X：几条*n
        cost,opt=self.sess.run([self.cost, self.optimizer],
                                  feed_dict={self.x:X, self.scale:self.training_scale})
        return cost

    def calc_cost(self, X):
        '''
        不训练，只计算cost
        :param X:
        :return:
        '''
        return self.sess.run(self.cost, feed_dict={self.x: X, self.scale: self.training_scale})

###########################################################################################
#数据预处理
def standard_scale(X_train, X_test): #<-----数据集预处理部分
    '''
    0均值，1标准差
    :param X_train:
    :param X_test:
    :return:
    '''
    # 根据预估的训练集的参数生成预处理器
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test  = preprocessor.transform(X_test)
    return X_train, X_test

def get_random_block_from_data(data,batch_size):
    '''
    随机取一个batch的数据
    :param data:
    :param batch_size:
    :return:
    '''
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index+batch_size)]

#主函数
#初始化模型数据，展示计算图,并开始数据计算
AGN_AC=AdditiveGaussianNoiseAutoencoder(n_input=784,n_hidden=200,transfer_function=tf.nn.softplus,optimizer=tf.train.AdamOptimizer(learning_rate=0.01),scale=0.01)
writer=tf.summary.FileWriter(logdir='logs',graph=AGN_AC.sess.graph)
writer.close()

#读取数据
mnist = input_data.read_data_sets('../Mnist_data/', one_hot=True)
X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)

n_samples = int(mnist.train.num_examples)  # 训练样本总数
training_epochs = 20  # 训练轮数，1轮等于n_samples/batch_size
batch_size = 128  # batch容量
display_step = 1  # 展示间隔


#训练
for epoch in range(training_epochs):
    avg_cost=0
    total_batch=input(n_samples/batch_size)
    for i in range(total_batch):
        batch_xs=get_random_block_from_data(X_train,batch_size)
        cost = AGN_AC.partial_fit(batch_xs)
        avg_cost += cost / batch_size
    avg_cost/=total_batch

    if epoch%display_step=='0':
        print('epoch:%04d,cost=%.9f'%(epoch,avg_cost))


# 计算测试集上的cost
print('Total coat:', str(AGN_AC.calc_cost(X_test)))













