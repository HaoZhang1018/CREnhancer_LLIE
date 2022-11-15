import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import initializers
from utils import *

def lrelu(x, trainbable=None):
    return tf.maximum(x*0.2,x)

def upsample_and_concat(x1, x2, output_channels, in_channels, scope_name, trainable=True):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        pool_size = 2
        deconv_filter = tf.get_variable('weights', [pool_size, pool_size, output_channels, in_channels], trainable= True)
        deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2) , strides=[1, pool_size, pool_size, 1], name=scope_name)

        deconv_output =  tf.concat([deconv, x2],3)
        deconv_output.set_shape([None, None, None, output_channels*2])

        return deconv_output



def DNet(input,training=True):
    with tf.variable_scope('DNet', reuse=tf.AUTO_REUSE):
        conv1=slim.conv2d(input,32,[3,3], rate=1, activation_fn=lrelu,scope='De_conv1')
        pool1=slim.max_pool2d(conv1, [2, 2], stride = 2, padding='SAME' )
        conv2=slim.conv2d(pool1,64,[3,3], rate=1, activation_fn=lrelu,scope='De_conv2')
        pool2=slim.max_pool2d(conv2, [2, 2], stride = 2, padding='SAME' )
        conv3=slim.conv2d(pool2,128,[3,3], rate=1, activation_fn=lrelu,scope='De_conv3')
        up1 =  upsample_and_concat( conv3, conv2, 64, 128 , 'De_up_1')
        conv4=slim.conv2d(up1,  64,[3,3], rate=1, activation_fn=lrelu,scope='De_conv4')
        up2 =  upsample_and_concat( conv4, conv1, 32, 64 , 'De_up_2')
        conv5=slim.conv2d(up2,  32,[3,3], rate=1, activation_fn=lrelu,scope='De_conv5')
                        
        r_conv2=slim.conv2d(conv5,8,[3,3], rate=1, activation_fn=lrelu,scope='R_conv2')
        r_conv3=slim.conv2d(r_conv2,3,[1,1], rate=1, activation_fn=None, scope='R_conv3')
        R_out = tf.sigmoid(r_conv3)

        l_conv2=slim.conv2d(conv1,32,[3,3], rate=1, activation_fn=lrelu,scope='L_conv2')
        l_conv_cat=tf.concat([l_conv2, conv5],3)
        l_conv3=slim.conv2d(l_conv_cat,32,[3,3], rate=1, activation_fn=lrelu,scope='L_conv3')
        l_conv4=slim.conv2d(l_conv3,16,[3,3], rate=1, activation_fn=lrelu, scope='L_conv4')
        l_conv5=slim.conv2d(l_conv4,8,[3,3], rate=1, activation_fn=lrelu, scope='L_conv5')
        l_conv6=slim.conv2d(l_conv5,1,[1,1], rate=1, activation_fn=None, scope='L_conv6')
        L_out = tf.sigmoid(l_conv6)

        L_out_3=tf.concat([L_out,L_out,L_out],axis=-1)
        D_out=input-(R_out*L_out)
        
    return R_out, L_out, D_out



def Enhance_net(input_low, training = True):
    with tf.variable_scope('Enhance_net', reuse=tf.AUTO_REUSE):
        conv1=slim.conv2d(input_low, 32,[3,3], rate=1, activation_fn=lrelu,scope='En_conv1_1')
        conv1=slim.conv2d(conv1,64,[3,3], rate=1, activation_fn=lrelu,scope='En_conv1_2')
        conv1_out = conv1

        conv2=slim.conv2d(conv1_out,128,[3,3], rate=1, activation_fn=lrelu,scope='En_conv2_1')
        conv2=slim.conv2d(conv2,256,[3,3], rate=1, activation_fn=lrelu,scope='En_conv2_2')
        conv2_out =conv2

        conv3=slim.conv2d(conv2_out,512,[3,3], rate=1, activation_fn=lrelu,scope='En_conv3_1')
        conv3=slim.conv2d(conv3,256,[3,3], rate=1, activation_fn=lrelu,scope='En_conv3_2')
        conv3_out =conv3

        conv4=slim.conv2d(conv3_out,128,[3,3], rate=1, activation_fn=lrelu,scope='En_conv4_1')
        conv4=slim.conv2d(conv4,64,[3,3], rate=1, activation_fn=lrelu,scope='En_conv4_2')
        conv4_out =conv4

        conv5=slim.conv2d(conv4_out,32,[3,3], rate=1, activation_fn=lrelu,scope='En_conv5_1')
        conv5=slim.conv2d(conv5,3,[3,3], rate=1, activation_fn=None, scope='En_conv5_2')
        conv5_out =tf.sigmoid(conv5)
        return conv5_out


def L_discriminator(input_light,training = True):
    with tf.variable_scope('L_discriminator', reuse=tf.AUTO_REUSE):
        conv1=slim.conv2d(input_light, 32,[3,3], rate=1, stride=2,padding='VALID',activation_fn=lrelu,scope='Dis_conv1')
        conv2=slim.conv2d(conv1,64,[3,3], rate=1, stride=2,padding='VALID',activation_fn=lrelu,scope='Dis_conv2')
        conv3=slim.conv2d(conv2,128,[3,3], rate=1, stride=2,padding='VALID',activation_fn=lrelu,scope='Dis_conv3')
        conv4=slim.conv2d(conv3,256,[3,3], rate=1, stride=2,padding='VALID',activation_fn=lrelu,scope='Dis_conv4')
        conv5 = tf.reshape(conv4,[10,2*2*256])
        conv6 = slim.fully_connected(conv5,1,activation_fn=None,scope='Dis_full')
        
        return conv6
