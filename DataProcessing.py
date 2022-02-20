# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 11:05:10 2022

@author: Tushar
"""
from email.policy import default
import json
from collections import namedtuple
import pandas as pd
import argparse
import os
import numpy as np
import io
from PIL import Image
#import tensorflow as tf
from PIL import Image
from object_detection.utils import dataset_util, label_map_util
import tensorflow.compat.v1 as tf



# Initiate argument parser
parser = argparse.ArgumentParser(
    description=" JSON-to-TFRecord converter")

parser.add_argument("-x",
                    "--json_dir",
                    help="Path to the folder where the input json files are stored.",
                    type=str, default=r'images/trainval/annotations/bbox-annotations.json')

parser.add_argument("-l",
                    "--labels_path",
                    help="Path to the labels (.pbtxt) file.", type=str, default=r'annotations/label_map.pbtxt')
parser.add_argument("-o",
                    "--output_path",
                    help="Path of output TFRecord (.record) file.", type=str, default=r'annotations')

parser.add_argument("-i",
                    "--image_dir",
                    help="Path to the folder where the input image files are stored. "
                         "Defaults to the same directory as XML_DIR.",
                    type=str, default=r'images/trainval/images')
parser.add_argument("-c",
                    "--csv_path",
                    help="Path of output .csv file. If none provided, then no file will be "
                         "written.",
                    type=str, default=None)

parser.add_argument("-r",
                    "--ratio",
                    help="ratio of train test split",
                    type=float, default=0.3)



args = parser.parse_args()



  
def json_to_csv(path):
    with open(path, 'r') as f:
      data = json.load(f)
    df = pd.DataFrame(data['annotations'])
    filenames = [image['file_name'] for image in data['images']]
    return df, filenames

def box_center_to_corner(box):
    """Convert from (center, width, height) to (upper-left, lower-right)."""
    cx, cy, w, h = box[0], box[1], box[2], box[3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    box = np.stack((x1, y1, x2, y2), axis=-1)
    return box

def preprocess(bbox):
    xmin, xmax, ymin, ymax = bbox[0], bbox[2], bbox[1], bbox[3]
    x1 = min(xmin, xmax)
    x2 = max(xmin, xmax)
    y1 = min(ymin, ymax)
    y2 = max(ymin, ymax)
    boxes = np.stack((x1, y1, x2, y2), axis=-1)
    return boxes




def split(df, group, filenames):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filenames[filename], gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['bbox'][0] / width)
        xmaxs.append(row['bbox'][2] / width)
        ymins.append(row['bbox'] [1]/ height)
        ymaxs.append(row['bbox'] [3]/ height)
        #classes_text.append(row['class'].encode('utf8'))
        classes.append(row['category_id'])
        if row['category_id'] == 0:
            class_txt = 'person'.encode('utf8')
            classes_text.append(class_txt)
            #print(class_txt)
        else:
            class_txt = 'car'.encode('utf8')
            #print(class_txt)
            classes_text.append(class_txt)    
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example




def main(_):
    output_path = os.path.join(os.getcwd(), args.output_path)
    json_path = os.path.join(os.getcwd(), args.json_dir)
    image_path = os.path.join(os.getcwd(), args.image_dir)
    examples, filenames = json_to_csv(json_path)
    examples['bbox'] = examples['bbox'].apply(preprocess)
    grouped = split(examples, 'image_id', filenames)
    ratio = int(args.ratio * len(grouped))
    group_train = grouped[:ratio]
    group_test = grouped[ratio:]
    grouped = [(group_train, 'train.record'), (group_test, 'test.record')]
    for data in grouped:
        fname = os.path.join(output_path, data[1])
        writer = tf.python_io.TFRecordWriter(fname)
            
        for group in data[0]:
            tf_example = create_tf_example(group, image_path)
            writer.write(tf_example.SerializeToString())
        writer.close()
    print('Successfully created the TFRecord file: {}'.format(output_path))
    csv_path = 'None'
    if csv_path is not None:
        examples.to_csv(csv_path, index=None)
        print('Successfully created the CSV file: {}'.format(csv_path))




if __name__ == '__main__':
    tf.app.run()        
