# coding: utf-8
from __future__ import print_function
import os
import time
import random
from PIL import Image
import tensorflow as tf
import numpy as np
from utils import *
from model import *
from glob import glob
from skimage import color,filters
import argparse

parser = argparse.ArgumentParser(description='')

parser.add_argument('--save_dir', dest='save_dir', default='./results/enhance/', help='directory for testing outputs')
parser.add_argument('--test_dir', dest='test_dir', default='./', help='directory for testing inputs')
args = parser.parse_args()

sess = tf.Session()
training = tf.placeholder_with_default(False, shape=(), name='training')
input_low_image = tf.placeholder(tf.float32, [None, None, None, 3], name='input_low')

[output_scene,output_light,output_degra] = DNet(input_low_image,training=False)

Enhance_image = Enhance_net(input_low_image,training=False)
[enhance_scene,enhance_light,enhance_degra] = DNet(Enhance_image,training=False)


# load pretrained model
var_Decom = [var for var in tf.trainable_variables() if 'DNet' in var.name]
var_Enhance = [var for var in tf.trainable_variables() if 'Enhance_net' in var.name]

g_list = tf.global_variables()

saver_Decom = tf.train.Saver(var_list = var_Decom)
saver_Enhance = tf.train.Saver(var_list = var_Enhance)


Decom_checkpoint_dir ='./checkpoint/DNet/'
ckpt_pre=tf.train.get_checkpoint_state(Decom_checkpoint_dir)
if ckpt_pre:
    print('loaded '+ckpt_pre.model_checkpoint_path)
    saver_Decom.restore(sess,ckpt_pre.model_checkpoint_path)
else:
    print('No DNet checkpoint!')

enhance_checkpoint_dir ='./checkpoint/Enhance_net/'
ckpt_pre_en=tf.train.get_checkpoint_state(enhance_checkpoint_dir)
if ckpt_pre_en:
    print('loaded '+ckpt_pre_en.model_checkpoint_path)
    saver_Enhance.restore(sess,ckpt_pre_en.model_checkpoint_path)
else:
    print('No Enhance_net checkpoint!')


save_dir = args.save_dir
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
    
    
###load eval data
eval_low_data = []
eval_img_name =[]
eval_low_data_name = glob(args.test_dir+'*')
eval_low_data_name.sort()
for idx in range(len(eval_low_data_name)):
    [_, name] = os.path.split(eval_low_data_name[idx])
    suffix = name[name.find('.') + 1:]
    name = name[:name.find('.')]
    eval_img_name.append(name)
    eval_low_im = load_images_no_norm(eval_low_data_name[idx])
    print(eval_low_im.shape)
    h,w,c = eval_low_im.shape
    h_tmp = h%1
    w_tmp = w%1
    eval_low_im_resize = eval_low_im[0:h-h_tmp, 0:w-w_tmp, :]
    print(eval_low_im_resize.shape)
    eval_low_data.append(eval_low_im_resize)


print("Start evalating!")
start_time = time.time()
for idx in range(len(eval_low_data)):
    print(idx)
    name = eval_img_name[idx]
    input_low = eval_low_data[idx]
    input_low_eval = np.expand_dims(input_low, axis=0)
    h, w, _ = input_low.shape
    
    decom_scene, decom_light = sess.run([output_scene, output_light], feed_dict={input_low_image:input_low_eval,training:False})
    output_enhance = sess.run(Enhance_image, feed_dict={input_low_image:input_low_eval,training:False})
            
    final_enhance = decom_light*input_low_eval + (1-decom_light)*output_enhance
    save_images(os.path.join(save_dir, '%s.png' % (name)), final_enhance)

    
    
