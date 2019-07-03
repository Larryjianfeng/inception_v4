import os
from nets.inception_v4 import * 
import tensorflow as tf
import numpy as np
import time
from PIL import Image
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

sess = tf.Session()
checkpoint_file = './inception_v4.ckpt'
sample_images = ['yi007.jpeg']
sess = tf.Session()

im_size = 299
inception_v4.default_image_size = im_size

arg_scope = inception_utils.inception_arg_scope()
inputs = tf.placeholder(tf.float32, (None, im_size, im_size, 3))

with slim.arg_scope(arg_scope):
    logits, end_points = inception_v4(inputs, is_training=False, dropout_keep_prob=1.0)
saver = tf.train.Saver()

saver.restore(sess, checkpoint_file)

stime = time.time()
for image in sample_images:
    im = Image.open(image)
    im = im.resize((299, 299))
    im = np.array(im)
    im = im.reshape(-1, 299, 299, 3)
    im = 2. * (im / 255.) - 1.
    logit_values, end_points = sess.run((logits, end_points), feed_dict={inputs: im})
    for k, v in end_points.items():
        print(k, v.shape)
    print(np.argmax(logit_values))
print('{} pictures takes {}'.format(len(sample_images), time.time()-stime))

