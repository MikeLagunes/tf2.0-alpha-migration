from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
#AUTOTUNE = tf.data.experimental.AUTOTUNE

import pathlib, os, glob, random

#import IPython.display as display

all_image_paths = glob.glob("/media/mikelf/rob/datasets/core50_v3/train/*")

data_root = pathlib.Path("/media/mikelf/rob/datasets/core50_v3/train/")
print(data_root)

#for item in data_root.iterdir():
#  print(item)


all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)

image_count = len(all_image_paths)
print (image_count)

print (all_image_paths[:10])

#path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
#print(path_ds)



for n in range(3):
  img_path = random.choice(all_image_paths)
  lbl = np.array([ int(img_path[-10:-8]) - 1])
  img = mpimg.imread(img_path)
  plt.imshow(img)
  plt.show()
  print (lbl)


img_raw = tf.io.read_file(img_path)

img_tensor = tf.image.decode_image(img_raw)

img_final = tf.image.resize(img_tensor, [192, 192])
img_final = img_final/255.0

print(img_final.shape)
print(img_final.numpy().min())
print(img_final.numpy().max())


def preprocess_image(image):
  image = tf.image.decode_image(image, channels=3)
  image = tf.image.resize(image, [192, 192])
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



