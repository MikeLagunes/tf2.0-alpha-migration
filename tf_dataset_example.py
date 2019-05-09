from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import time
#AUTOTUNE = tf.data.experimental.AUTOTUNE

import pathlib, os, glob, random
AUTOTUNE = tf.data.experimental.AUTOTUNE

#import IPython.display as display

all_image_paths = glob.glob("/media/mikelf/rob/datasets/core50_v3/train/*")

data_root = pathlib.Path("/media/mikelf/rob/datasets/core50_v3/train/")
#print(data_root)

#for item in data_root.iterdir():
#  print(item)

# Get all image filenames

all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]

all_image_paths = all_image_paths#[0:10000]
#random.shuffle(all_image_paths)

image_count = len(all_image_paths)


def get_labels(all_img_paths):
  
  lbls = []

  for img_path in all_img_paths:
    lbl = np.array([ int(img_path[-10:-8]) - 1])  # normalize to [0,1] range
    lbls.append(lbl)

  return lbls
#print (image_count)


def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, size=[192, 192])
  image /= 255.0  # normalize to [0,1] range

  return image

def load_and_preprocess_image(img_path):
  print(img_path)
  image = tf.io.read_file(img_path)

  #lbl = np.array([ int(img_path[-10:-8]) - 1])
  #lbl = tf.cast(lbl, tf.int64)
  return preprocess_image(image)


def visualize_image(img_path):
  lbl = np.array([ int(img_path[-10:-8]) - 1])
  plt.imshow(load_and_preprocess_image(img_path))
  plt.grid(False)
  plt.xlabel("Object {}".format(lbl + 1))
  print()

def load_and_preprocess_from_path_label(path, label):
  return load_and_preprocess_image(path), label

#print(all_image_paths)


#img_path = all_image_paths[0]

# plt.imshow(load_and_preprocess_image(img_path))
# plt.grid(False)
# plt.show()

#path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)

all_image_labels = get_labels(all_image_paths)

ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))
image_label_ds = ds.map(load_and_preprocess_from_path_label)
print(image_label_ds)



#image_ds = path_ds.map(load_and_preprocess_image)
#label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))


#print(get_labels(all_image_paths)[0:3],all_image_paths[0:3] )
#image_ds = path_ds.map(load_and_preprocess_image)

plt.figure(figsize=(8,8))
for n,(image, lbl) in enumerate(image_label_ds.take(4)):
  plt.subplot(2,2,n+1)
  plt.imshow(image)
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  plt.xlabel(lbl)

plt.show()


BATCH_SIZE = 32

# Setting a shuffle buffer size as large as the dataset ensures that the data is
# completely shuffled.

image_label_ds = image_label_ds.cache(filename='./cache.tf-data')


ds = image_label_ds.shuffle(buffer_size=image_count)
ds = ds.repeat()
ds = ds.batch(BATCH_SIZE)
# `prefetch` lets the dataset fetch batches, in the background while the model is training.
ds = ds.prefetch(buffer_size=AUTOTUNE)



mobile_net = tf.keras.applications.MobileNetV2(input_shape=(192, 192, 3), include_top=False)
mobile_net.trainable=False


def change_range(image,label):
  return 2*image-1, label

keras_ds = ds.map(change_range)

image_batch, label_batch = next(iter(keras_ds))

feature_map_batch = mobile_net(image_batch)

print(feature_map_batch.shape)


steps_per_epoch=tf.math.ceil(len(all_image_paths)/BATCH_SIZE).numpy()
default_timeit_steps = 2*steps_per_epoch+1

def timeit(ds, steps=default_timeit_steps):
  overall_start = time.time()
  # Fetch a single batch to prime the pipeline (fill the shuffle buffer),
  # before starting the timer
  it = iter(ds.take(steps+1))
  next(it)

  start = time.time()
  for i,(images,labels) in enumerate(it):
    if i%10 == 0:
      print('.',end='')
  print()
  end = time.time()

  duration = end-start
  print("{} batches: {} s".format(steps, duration))
  print("{:0.5f} Images/s".format(BATCH_SIZE*steps/duration))
  print("Total time: {}s".format(end-overall_start))


timeit(ds)










