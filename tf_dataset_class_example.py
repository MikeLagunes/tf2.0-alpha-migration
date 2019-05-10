from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import time
import pathlib, os, glob, random
from random import shuffle

AUTOTUNE = tf.data.experimental.AUTOTUNE

image_size = []

def get_filenames(rootdir='', shuffe=''):
  print(rootdir)
  data_root = pathlib.Path(rootdir)
  all_image_paths = list(data_root.glob('*/*'))
  all_image_paths = [str(path) for path in all_image_paths]

  if shuffle:
    shuffle(all_image_paths)# Repeat shuffle to increase randomeness

  return all_image_paths


def get_labels(all_img_paths):
  
  lbls = []

  for img_path in all_img_paths:
    lbl = np.array([ int(img_path[-10:-8]) - 1])  # normalize to [0,1] range
    lbls.append(lbl)

  return lbls


def preprocess_image(image):
  global image_size
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, size=image_size)
  image /= 255.0  # normalize to [0,1] range

  return image

def load_and_preprocess_image(img_path):
  image = tf.io.read_file(img_path)
  return preprocess_image(image)


def visualize_image(img_path):
  lbl = np.array([ int(img_path[-10:-8]) - 1])
  plt.imshow(load_and_preprocess_image(img_path))
  plt.grid(False)
  plt.xlabel("Object {}".format(lbl + 1))
  print()

def load_and_preprocess_from_path_label(path, label):
  return load_and_preprocess_image(path), label

class core50():
  """ on-the-fly loader for CORe50 dataset 
  """

  def __init__(self, root, split="train", img_size=[192, 192], augmentations=None, instances=None, shuffle=False, batch_size=32):

    """__init__
    """

    global image_size
    image_size = img_size

    self.augmentations = augmentations
    self.num_classes = 50
    self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
    self.files = {}
    self.images_base = os.path.join(root, split)
    self.files[split] = get_filenames(rootdir=self.images_base, shuffe=shuffle)
    self.image_count = len(self.files[split])
    image_size = img_size
    labels = get_labels(self.files[split])
    ds = tf.data.Dataset.from_tensor_slices((self.files[split], labels))

    self.image_label_ds = ds.map(load_and_preprocess_from_path_label)
    self.image_label_ds = self.image_label_ds.batch(batch_size).prefetch(buffer_size=AUTOTUNE)



if __name__ == '__main__':

  local_path = '/media/mikelf/rob/datasets/core50_v3/'
  ds = core50(local_path, augmentations=None)

  print(ds.image_label_ds)




