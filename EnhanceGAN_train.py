# coding: utf-8
from __future__ import print_function

import os
import time
import random
from skimage import color
from PIL import Image
import tensorflow as tf
import numpy as np

from utils import *
from model import *
from glob import glob

training = tf.placeholder_with_default(False, shape=(), name='training')
batch_size = 10
patch_size = 48
learning_rate=1e-4


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)

#enhance input
input_low_to_enhance = tf.placeholder(tf.float32, [None, None, None, 3], name='input_low_to_enhance')

#well light reference
input_well_reference = tf.placeholder(tf.float32, [None, None, None, 3], name='input_well_reference')

[R_decom_low, S_decom_low, D_decom_low] = DNet(input_low_to_enhance, training=False)

#Enhance
output_Enhance = Enhance_net(input_low_to_enhance)


[R_decom_enhance, S_decom_enhance, D_decom_enhance] = DNet(output_Enhance, training=False)
[R_decom_well, S_decom_well, D_decom_well] = DNet(input_well_reference, training=False)

#discriminator
True_prob=L_discriminator(S_decom_well)
Fake_prob=L_discriminator(S_decom_enhance)

## define loss

#discriminator loss
Dis_loss_true=tf.reduce_mean(tf.abs(True_prob-tf.random_uniform(shape=[batch_size,1],minval=0.8,maxval=1.0,dtype=tf.float32)))
Dis_loss_fake=tf.reduce_mean(tf.abs(Fake_prob-tf.random_uniform(shape=[batch_size,1],minval=0.0,maxval=0.2,dtype=tf.float32)))
Dis_loss_total=Dis_loss_true+Dis_loss_fake
tf.summary.scalar('Dis_loss_total',Dis_loss_total)



#Enhancer loss

#Scene fidelity loss
R_decom_low_1 = R_decom_low[:,:,:,0:1]
R_decom_enhance_1 = R_decom_enhance[:,:,:,0:1]
ssim_r_1 = tf_ssim(R_decom_low_1, R_decom_enhance_1)
R_decom_low_2 = R_decom_low[:,:,:,1:2]
R_decom_enhance_2 = R_decom_enhance[:,:,:,1:2]
ssim_r_2 = tf_ssim(R_decom_low_2, R_decom_enhance_2)
R_decom_low_3 = R_decom_low[:,:,:,2:3]
R_decom_enhance_3 = R_decom_enhance[:,:,:,2:3]
ssim_r_3 = tf_ssim(R_decom_low_3, R_decom_enhance_3)
ssim_r = (ssim_r_1 + ssim_r_2 + ssim_r_3)/3.0
loss_sf = 1-ssim_r


#Degradation Suppression loss
def D_int_loss(input_D, input_S):
    input_D_low_1=tf.expand_dims(input_D[:,:,:,0],-1)
    D_loss_1 = tf.abs(tf.div(input_D_low_1, tf.maximum(1-input_S, 0.01)))    
    
    input_D_low_2=tf.expand_dims(input_D[:,:,:,1],-1)
    D_loss_2 = tf.abs(tf.div(input_D_low_2, tf.maximum(1-input_S, 0.01)))   
    
    input_D_low_3=tf.expand_dims(input_D[:,:,:,2],-1)
    D_loss_3 = tf.abs(tf.div(input_D_low_3, tf.maximum(1-input_S, 0.01))) 
    
    mut_D_loss = tf.reduce_mean((D_loss_1+D_loss_2+D_loss_3)/3.0) 
    return mut_D_loss

def Grad_D_com(input_D):
    input_D1=tf.expand_dims(input_D[:,:,:,0],-1)
    D1_gradient_x = gradient(input_D1, "x")
    D1_gradient_y = gradient(input_D1, "y")
    D1_gradient=tf.abs(D1_gradient_x)+tf.abs(D1_gradient_y)
    
    input_D2=tf.expand_dims(input_D[:,:,:,1],-1)
    D2_gradient_x = gradient(input_D2, "x")
    D2_gradient_y = gradient(input_D2, "y")
    D2_gradient=tf.abs(D2_gradient_x)+tf.abs(D2_gradient_y)

    input_D3=tf.expand_dims(input_D[:,:,:,2],-1)
    D3_gradient_x = gradient(input_D3, "x")
    D3_gradient_y = gradient(input_D3, "y")
    D3_gradient=tf.abs(D3_gradient_x)+tf.abs(D3_gradient_y)
    
    D_gradient_total=(D1_gradient+D2_gradient+D3_gradient)/3;
    return D_gradient_total
    
D_loss_suppres_int=D_int_loss(D_decom_enhance,S_decom_enhance)
D_loss_suppres_grad=tf.reduce_mean(Grad_D_com(D_decom_enhance))
D_loss_suppres = 0.01*D_loss_suppres_int + 1 * D_loss_suppres_grad

#Illumination adversarial loss
enhance_loss_dis=tf.reduce_mean(tf.abs(Fake_prob-tf.random_uniform(shape=[batch_size,1],minval=0.8,maxval=1.0,dtype=tf.float32)))
#Enhance total loss
Enhance_loss_total =  1*loss_sf+0.3*enhance_loss_dis+10*D_loss_suppres



tf.summary.scalar('loss_sf',loss_sf)
tf.summary.scalar('D_loss_suppres_int',D_loss_suppres_int)
tf.summary.scalar('D_loss_suppres_grad',D_loss_suppres_grad)
tf.summary.scalar('enhance_loss_dis',enhance_loss_dis)
tf.summary.scalar('Enhance_loss_total',Enhance_loss_total)

tf.summary.image('input_low_to_enhance',tf.expand_dims(input_low_to_enhance[1,:,:,:],0))
tf.summary.image('input_well_reference',tf.expand_dims(input_well_reference[1,:,:,:],0))
tf.summary.image('D_decom_enhance',tf.expand_dims(D_decom_enhance[1,:,:,:],0)) 
tf.summary.image('D_decom_low',tf.expand_dims(D_decom_low[1,:,:,:],0))
tf.summary.image('R_decom_low',tf.expand_dims(R_decom_low[1,:,:,:],0))  
tf.summary.image('R_decom_enhance',tf.expand_dims(R_decom_enhance[1,:,:,:],0)) 
tf.summary.image('S_decom_enhance',tf.expand_dims(S_decom_enhance[1,:,:,:],0))
tf.summary.image('S_decom_low',tf.expand_dims(S_decom_low[1,:,:,:],0))
tf.summary.image('S_decom_well',tf.expand_dims(S_decom_well[1,:,:,:],0))        
tf.summary.image('output_Enhance',tf.expand_dims(output_Enhance[1,:,:,:],0))
summary_op = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('./log/' + '/EnhanceGAN',sess.graph,flush_secs=60)


var_Decom = [var for var in tf.trainable_variables() if 'DNet' in var.name]
var_enhance = [var for var in tf.trainable_variables() if 'Enhance_net' in var.name]
var_dis = [var for var in tf.trainable_variables() if 'L_discriminator' in var.name]
g_list = tf.global_variables()


with tf.name_scope('train_step'):
    train_enhance_op = tf.train.AdamOptimizer(learning_rate).minimize(Enhance_loss_total,var_list=var_enhance)
    train_discriminator_op=tf.train.AdamOptimizer(learning_rate).minimize(Dis_loss_total,var_list=var_dis)
   

saver_enhance = tf.train.Saver(var_list=var_enhance,max_to_keep=2000)
saver_Dis = tf.train.Saver(var_list=var_dis,max_to_keep=2000)
saver_Decom = tf.train.Saver(var_list = var_Decom,max_to_keep=2000)
sess.run(tf.global_variables_initializer())
print("[*] Initialize model successfully...")


pre_checkpoint_dir = './checkpoint/DNet/'
ckpt_pre=tf.train.get_checkpoint_state(pre_checkpoint_dir)
if ckpt_pre:
    print('loaded '+ckpt_pre.model_checkpoint_path)
    saver_Decom.restore(sess,ckpt_pre.model_checkpoint_path)
else:
    print('No DNet pre_checkpoint!')


eval_low_data = []
eval_high_data = []

eval_low_data_name =  glob('./low/*.png')
eval_low_data_name.sort()
for idx in range(len(eval_low_data_name)):
    eval_low_im = load_images_no_norm(eval_low_data_name[idx])
    eval_low_data.append(eval_low_im)

eval_high_data_name =  glob('./well/*.png')
eval_high_data_name.sort()
for idx in range(len(eval_high_data_name)):
    eval_high_im = load_images_no_norm(eval_high_data_name[idx])
    eval_high_data.append(eval_high_im)

epoch = 600
numBatch = len(eval_high_data) // int(batch_size)
enhance_checkpoint_dir = './checkpoint/Enhance_net/'
if not os.path.isdir(enhance_checkpoint_dir):
    os.makedirs(enhance_checkpoint_dir)



start_step = 0
start_epoch = 0
iter_num = 0

start_time = time.time()
image_id = 0
counter=0
for epoch in range(start_epoch, epoch):
    for batch_id in range(start_step, numBatch):
        batch_input_low = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
        batch_input_well = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")


        for patch_id in range(batch_size):
            h, w, _ = eval_high_data[image_id].shape
            x = random.randint(0, h - patch_size)
            y = random.randint(0, w - patch_size)
            i_low_expand = np.expand_dims(eval_high_data[image_id], axis = 2)
            rand_mode = random.randint(0, 7)
            batch_input_low[patch_id, :, :, :] =  data_augmentation(eval_low_data[image_id] [x : x+patch_size, y : y+patch_size, :] , rand_mode)
            batch_input_well[patch_id, :, :, :] = data_augmentation(eval_high_data[image_id][x : x+patch_size, y : y+patch_size, :] , rand_mode)

            image_id = (image_id + 1) % len(eval_low_data)
            if image_id == 0:
                tmp = list(zip(eval_low_data, eval_high_data))
                random.shuffle(list(tmp))
                eval_low_data, eval_high_data= zip(*tmp)

        for i in range(1):
            _, loss_dis = sess.run([train_discriminator_op, Dis_loss_total], feed_dict={input_low_to_enhance: batch_input_low,input_well_reference: batch_input_well})
        _, loss_enhance,summary_str= sess.run([train_enhance_op, Enhance_loss_total,summary_op], feed_dict={input_low_to_enhance: batch_input_low,input_well_reference: batch_input_well})
        if counter % 10 == 0:
            print("Epoch: [%2d], step: [%2d],  loss_d: [%.8f],loss_g:[%.8f]" \
              % ((epoch+1), iter_num,  loss_dis,loss_enhance))
        train_writer.add_summary(summary_str,iter_num)
        iter_num += 1

        
    global_step = epoch+1
    if (epoch+1)%10==0:
      saver_enhance.save(sess, enhance_checkpoint_dir + 'model.ckpt', global_step=global_step)
      #saver_Dis.save(sess, dis_checkpoint_dir + 'model.ckpt', global_step=global_step)

print("[*] Finish training")






