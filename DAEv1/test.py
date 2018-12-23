#encoding:utf-8
import tensorflow as tf
if __name__=='__main__':
   with tf.variable_scope("foo"):
        # 在命名空间foo下获取变量"bar"，于是得到的变量名称为"foo/bar"。
        a = tf.get_variable("bar",[1,2,3])   #获取变量名称为“bar”的变量
        print(a)
   with tf.variable_scope("bar"):
        # 在命名空间bar下获取变量"bar"，于是得到的变量名称为"bar/bar"。
        a = tf.get_variable("bar",[1])
        print(a.name)
   with tf.name_scope("a"):
        # 使用tf.Variable函数生成变量会受tf.name_scope影响，于是得到的变量名称为"a/Variable"。
        a = tf.Variable([1])
        #新建变量
        print(a.name)
        #输出:a/Variable:0

        # 使用tf.get_variable函数生成变量不受tf.name_scope影响，于是变量并不在a这个命名空间中。
        a = tf.get_variable("b",[1])
        print(a.name)
        #输出:b:0
   with tf.name_scope("b"):
        # 使用tf.get_variable函数生成变量不受tf.name_scope影响，所以这里将试图获取名称
        # 为“b”的变量。然而这个变量已经被声明了，于是这里会报重复声明的错误
        tf.get_variable("b",[1])#提示错误
