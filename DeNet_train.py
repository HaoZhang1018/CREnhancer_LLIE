# coding: utf-8
from __future__ import print_function
import os, time, random
import tensorflow as tf
from PIL import Image
import numpy as np
from utils import *
from model import *
from glob import glob
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=10, help='number of samples in one batch')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=48, help='patch size')
parser.add_argument('--train_data_dir', dest='train_data_dir', default='./', help='directory for training inputs')


args = parser.parse_args()

batch_size = args.batch_size
patch_size = args.patch_size

sess = tf.Session()

input_low = tf.placeholder(tf.float32, [None, None, None, 3], name='input_low')
input_high = tf.placeholder(tf.float32, [None, None, None, 3], name='input_high')

[R_low, S_low,D_low] = DNet(input_low)
[R_high, S_high,D_high] = DNet(input_high)

S_low_3=tf.concat([S_low,S_low,S_low],axis=-1)
S_high_3=tf.concat([S_high,S_high,S_high],axis=-1)
R_low_3 = R_low
R_high_3 = R_high
D_low_3 = D_low
D_high_3 = D_high

#LOSS FUNCTION


#REFLECTANCE CONSISTANCE LOSS
def S_smooth_loss(input_S_low, input_im):
    input_gray = tf.image.rgb_to_grayscale(input_im)
    low_gradient_x = gradient(input_S_low, "x")
    input_gradient_x = gradient(input_gray, "x")
    x_loss = tf.abs(tf.div(low_gradient_x, tf.maximum(input_gradient_x, 0.01)))
    low_gradient_y = gradient(input_S_low, "y")
    input_gradient_y = gradient(input_gray, "y")
    y_loss = tf.abs(tf.div(low_gradient_y, tf.maximum(input_gradient_y, 0.01)))
    mut_loss = tf.reduce_mean(x_loss + y_loss) 
    return mut_loss

S_smooth_loss_high = S_smooth_loss(S_high, input_high)
S_smooth_loss_low  = S_smooth_loss(S_low, input_low)
S_smooth_loss=S_smooth_loss_high + S_smooth_loss_low  


#REFLECTANCE CONSISTANCE LOSS
R_consis_loss=tf.reduce_mean(tf.abs(R_low_3-R_high_3))


#REFLECTANCE Regularization LOSS with different penalties
D_loss_regular_low=tf.reduce_mean(tf.square(D_low_3))
D_loss_regular_high=tf.reduce_mean(tf.square(D_high_3))
D_loss_regular=1*D_loss_regular_low+100*D_loss_regular_high


loss_DNet = 0.5 * R_consis_loss + 0.1* S_smooth_loss+ 1*D_loss_regular

tf.summary.scalar('R_consis_loss',R_consis_loss)
tf.summary.scalar('S_smooth_loss_high',S_smooth_loss_high)
tf.summary.scalar('S_smooth_loss_low',S_smooth_loss_low)
tf.summary.scalar('D_loss_regular_low',D_loss_regular_low)
tf.summary.scalar('D_loss_regular_high',D_loss_regular_high)
tf.summary.scalar('loss_DNet',loss_DNet)
###
lr = tf.placeholder(tf.float32, name='learning_rate')

optimizer = tf.train.AdamOptimizer(learning_rate=lr, name='AdamOptimizer')
var_Decom = [var for var in tf.trainable_variables() if 'DNet' in var.name]
train_op_Decom = optimizer.minimize(loss_DNet, var_list = var_Decom)
sess.run(tf.global_variables_initializer())
saver_Decom = tf.train.Saver(var_list = var_Decom,max_to_keep=1000)
print("[*] Initialize model successfully...")

with tf.name_scope('image'):
  tf.summary.image('input_low',tf.expand_dims(input_low[1,:,:,:],0))
  tf.summary.image('input_high',tf.expand_dims(input_high[1,:,:,:],0))
  tf.summary.image('R_low_3',tf.expand_dims(R_low_3[1,:,:,:],0))  
  tf.summary.image('R_high_3',tf.expand_dims(R_high_3[1,:,:,:],0)) 
  tf.summary.image('D_low_3',tf.expand_dims(D_low_3[1,:,:,:],0))
  tf.summary.image('D_high_3',tf.expand_dims(D_high_3[1,:,:,:],0))        
  tf.summary.image('S_low_3',tf.expand_dims(S_low_3[1,:,:,:],0))
  tf.summary.image('S_high_3',tf.expand_dims(S_high_3[1,:,:,:],0))
summary_op = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('./log' + '/DNet_train',sess.graph,flush_secs=60)
#load data
###train_data
train_low_data = []
train_high_data = []
train_low_data_names = glob(args.train_data_dir +'/low/*.png') 
train_low_data_names.sort()
train_high_data_names = glob(args.train_data_dir +'/high/*.png') 
train_high_data_names.sort()
assert len(train_low_data_names) == len(train_high_data_names)
print('[*] Number of training data: %d' % len(train_low_data_names))
for idx in range(len(train_low_data_names)):
    low_im = load_images_no_norm(train_low_data_names[idx])
    train_low_data.append(low_im)
    high_im = load_images_no_norm(train_high_data_names[idx])
    train_high_data.append(high_im)


epoch = 4000
learning_rate = 0.0001

train_phase = 'DNet'
numBatch = len(train_low_data) // int(batch_size)
train_op = train_op_Decom
train_loss = loss_DNet
saver = saver_Decom

checkpoint_dir = './checkpoint/DNet/'
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)
ckpt=tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
    print('loaded '+ckpt.model_checkpoint_path)
    saver.restore(sess,ckpt.model_checkpoint_path)
else:
    print('No DNet pretrained model!')

start_step = 0
start_epoch = 0
iter_num = 0

print("[*] Start training for phase %s, with start epoch %d start iter %d : " % (train_phase, start_epoch, iter_num))
start_time = time.time()
image_id = 0
counter = 0
for epoch in range(start_epoch, epoch):
    for batch_id in range(start_step, numBatch):
        batch_input_low = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
        batch_input_high = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
        for patch_id in range(batch_size):
            h, w, _ = train_low_data[image_id].shape
            x = random.randint(0, h - patch_size)
            y = random.randint(0, w - patch_size)
            rand_mode = random.randint(0, 7)
            batch_input_low[patch_id, :, :, :] = data_augmentation(train_low_data[image_id][x : x+patch_size, y : y+patch_size, :], rand_mode)
            batch_input_high[patch_id, :, :, :] = data_augmentation(train_high_data[image_id][x : x+patch_size, y : y+patch_size, :], rand_mode)
            image_id = (image_id + 1) % len(train_low_data)
            if image_id == 0:
                tmp = list(zip(train_low_data, train_high_data))
                random.shuffle(tmp)
                train_low_data, train_high_data  = zip(*tmp)
        counter += 1
        _, loss,summary_str = sess.run([train_op, train_loss,summary_op], feed_dict={input_low: batch_input_low, \
                                                              input_high: batch_input_high, \
                                                              lr: learning_rate})
        print("%s Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f" \
              % (train_phase, epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
        train_writer.add_summary(summary_str,counter)
        iter_num += 1

    if (epoch+1)%10==0:     
      saver.save(sess, checkpoint_dir + 'model.ckpt',global_step=epoch+1)

print("[*] Finish training for phase %s." % train_phase)
