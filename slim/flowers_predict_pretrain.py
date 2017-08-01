import tensorflow as tf
from datasets import dataset_utils
from datasets import flowers
from tensorflow.contrib import slim
import numpy as np
from preprocessing import inception_preprocessing
from nets import inception
from datasets import imagenet

try:
    import urllib2 as urllib
except ImportError:
    import urllib.request as urllib
import os

url = "http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz"

checkpoints_dir = '/tmp/inceptionv1'

# if not tf.gfile.Exists(checkpoints_dir):
#     tf.gfile.MakeDirs(checkpoints_dir)
#
# dataset_utils.download_and_uncompress_tarball(url, checkpoints_dir)

image_size = inception.inception_v1.default_image_size


with tf.Graph().as_default():
    tf.logging.set_verbosity(tf.logging.INFO)
    #imageurl = 'https://upload.wikimedia.org/wikipedia/commons/7/70/EnglishCockerSpaniel_simon.jpg'
    imageurl = 'http://pic40.nipic.com/20140425/18581947_185648619125_2.jpg'
    image_string = urllib.urlopen(imageurl).read()
    image = tf.image.decode_jpeg(image_string,  channels=3)
    processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size,  is_training=False)
    processed_images = tf.expand_dims(processed_image, 0)

    with slim.arg_scope(inception.inception_v1_arg_scope()):
        logits, _ = inception.inception_v1(processed_images,num_classes=1001, is_training=False)
    probabilities = tf.nn.softmax(logits)

    init_fn = slim.assign_from_checkpoint_fn(
        os.path.join(checkpoints_dir, 'inception_v1.ckpt'),
        slim.get_model_variables('InceptionV1'))


    with tf.Session() as sess:
        init_fn(sess)
        np_image, probabilities = sess.run([image, probabilities])
        probabilities = probabilities[0, 0:]
        sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x:x[1])]

    names = imagenet.create_readable_names_for_imagenet_labels()
    for i in range(5):
        index = sorted_inds[i]
        print('Probability %0.2f%% => [%s]' % (probabilities[index] * 100, names[index]))
