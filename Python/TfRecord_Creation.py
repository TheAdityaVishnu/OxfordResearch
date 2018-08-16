import math
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import slim

from image2tfrecords.image2tfrecords import Image2TFRecords
from image2tfrecords.imagedataset import ImageDataSet

TUM_DATA_DIR = "/tmp/TUM_data/training"
TUM_LABELS = "/tmp/TUM_data/trainingLabels.csv"

img2tf = Image2TFRecords(
            TUM_DATA_DIR,
            TUM_LABELS,
            val_size=0.3,
            test_size=0.1,
            dataset_name="TUM"
            )
    img2tf.create_tfrecords(output_dir="/tmp/TUM_data/tfrecords")
