import tensorflow.compat.v1 as tf
import numpy as np
import os
import h5py
import sys
import argparse

sys.path.append('./utils')
sys.path.append('./models')

tf.disable_v2_behavior()

import dataset_human as dataset
import model_shapeV2 as model

parser = argparse.ArgumentParser()
#FLAGS = tf.compat.v1.app.flags.FLAGS
#tf.compat.v1.app.flags.DEFINE_string('train_dir', './train_shape_human',
#                           """Directory where to write summaries and checkpoint.""")
parser.add_argument('--train_dir', default='./train_shape_human', help= """Directory where to write summaries and checkpoint.""")
#tf.compat.v1.app.flags.DEFINE_string('base_dir', './data/human_im2avatar', 
#                           """The path containing all the samples.""")
parser.add_argument('--base_dir', default='./data/human_im2avatar', help="""The path containing all the samples.""")
#tf.compat.v1.app.flags.DEFINE_string('data_list_path', './data_list', 
#                          """The path containing data lists.""")
parser.add_argument('--data_list_path', default='./data_list', help= """The path containing data lists.""")
#tf.compat.v1.app.flags.DEFINE_string('output_dir', './output_shape_human',
#                           """Directory to save generated volume.""")
parser.add_argument('--output_dir', default='./output_shape_human', help= """Directory to save generated volume.""")

FLAGS = parser.parse_args()

TRAIN_DIR = FLAGS.train_dir
OUTPUT_DIR = FLAGS.output_dir

if not os.path.exists(OUTPUT_DIR): 
  os.makedirs(OUTPUT_DIR)

BATCH_SIZE = 12
IM_DIM = 128 
VOL_DIM = 64

def inference(dataset_):
  is_train_pl = tf.compat.v1.placeholder(tf.bool)
  img_pl, _, = model.placeholder_inputs(BATCH_SIZE, IM_DIM, VOL_DIM)
  pred = model.get_model(img_pl, is_train_pl)
  pred = tf.sigmoid(pred)

  config = tf.compat.v1.ConfigProto()
  config.gpu_options.allocator_type = 'BFC'
  config.gpu_options.allow_growth = True
  config.allow_soft_placement = True

  with tf.compat.v1.Session(config=config) as sess:
    model_path = os.path.join(TRAIN_DIR, "trained_models")
    ckpt = tf.train.get_checkpoint_state(model_path)
    restorer = tf.compat.v1.train.Saver()
    restorer.restore(sess, ckpt.model_checkpoint_path)

    test_samples = dataset_.getTestSampleSize()

    for batch_idx in range(test_samples):
      imgs, view_names = dataset_.next_test_batch(batch_idx, 1)  

      feed_dict = {img_pl: imgs, is_train_pl: False}
      pred_res = sess.run(pred, feed_dict=feed_dict)

      for i in range(len(view_names)):
        vol_ = pred_res[i]

        cloth = view_names[i][0]
        mesh = view_names[i][1]
        name_ = view_names[i][2][:-4]

        save_path = os.path.join(OUTPUT_DIR, cloth, mesh)
        if not os.path.exists(save_path): 
          os.makedirs(save_path)

        save_path_name = os.path.join(save_path, name_+".h5")
        if os.path.exists(save_path_name):
          os.remove(save_path_name)

        h5_fout = h5py.File(save_path_name)
        h5_fout.create_dataset(
                'data', data=vol_,
                compression='gzip', compression_opts=4,
                dtype='float32')
        h5_fout.close()

        print (batch_idx, save_path_name)

def main():
  test_dataset = dataset.Dataset(base_path=FLAGS.base_dir, 
                                 data_list_path=FLAGS.data_list_path)
  inference(test_dataset)

if __name__ == '__main__':
  main()
