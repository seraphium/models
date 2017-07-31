import tensorflow as tf
from datasets import dataset_utils
from datasets import flowers
from tensorflow.contrib import slim

flowers_data_dir = "/home/jackie/dev/data/flowers"

#download
# url = "http://download.tensorflow.org/data/flowers.tar.gz"
# if not tf.gfile.Exists(flowers_data_dir):
#     tf.gfile.MakeDirs(flowers_data_dir)
#
# dataset_utils.download_and_uncompress_tarball(url, flowers_data_dir)

with tf.Graph().as_default():
    datasets = flowers.get_split('train', flowers_data_dir)
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        datasets, common_queue_capacity=32, common_queue_min=1)
    image, label = data_provider.get(['image', 'label'])

    with tf.Session() as sess:
        with slim.queues.QueueRunners(sess):
            for i in range(4):
                np_image, np_label = sess.run([image, label])
                height, width, _ = np_image.shape
                class_name = name = datasets.labels_to_names[np_label]
                print(class_name)
