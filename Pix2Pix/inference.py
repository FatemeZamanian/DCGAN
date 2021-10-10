import argparse
import pathlib
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import cv2
import numpy as np

path=pathlib.Path('input')

parser = argparse.ArgumentParser()
parser.add_argument("--input", default=path, type=pathlib.Path)
args = parser.parse_args()

model=load_model('bw2colorfull2.h5')

out_images=[]
for i,p in enumerate(os.listdir(args.input)):
    image = tf.io.read_file(str(args.input/p))
    image = tf.image.decode_jpeg(image)
    image=tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    image = tf.image.resize(image, [256, 256],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image=tf.expand_dims(image,axis=0)

    pred=model(image,training=True)
    pred = pred * 0.5+0.5
    pred=tf.image.convert_image_dtype(pred,tf.uint8)
    pred=tf.squeeze(pred,axis=0)
    pred=np.array(pred)
    pred=cv2.cvtColor(pred,cv2.COLOR_BGR2RGB)
    cv2.imwrite(f'output/{i}.jpg',pred)

    




