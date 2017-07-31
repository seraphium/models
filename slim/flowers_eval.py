import tensorflow as tf
from datasets import dataset_utils
from datasets import flowers
from tensorflow.contrib import slim
import numpy as np
from preprocessing import inception_preprocessing

flowers_data_dir = "/home/jackie/dev/data/flowers"

#download
# url = "http://download.tensorflow.org/data/flowers.tar.gz"
# if not tf.gfile.Exists(flowers_data_dir):
#     tf.gfile.MakeDirs(flowers_data_dir)
#
# dataset_utils.download_and_uncompress_tarball(url, flowers_data_dir)

# with tf.Graph().as_default():
#     datasets = flowers.get_split('train', flowers_data_dir)
#     data_provider = slim.dataset_data_provider.DatasetDataProvider(
#         datasets, common_queue_capacity=32, common_queue_min=1)
#
#     with tf.Session() as sess:
#         with slim.queues.QueueRunners(sess):
#             for i in range(4):
#                 image, label = data_provider.get(['image', 'label'])
#
#                 np_image, np_label = sess.run([image, label])
#                 height, width, _ = np_image.shape
#                 class_name = name = datasets.labels_to_names[np_label]
#                 print(class_name)

#network defining
def my_cnn(images, num_classes, is_training):  # is_training is not used...
    with slim.arg_scope([slim.max_pool2d], kernel_size=[3, 3], stride=2):
        net = slim.conv2d(images, 64, [5, 5])
        net = slim.max_pool2d(net)
        net = slim.conv2d(net, 64, [5, 5])
        net = slim.max_pool2d(net)
        net = slim.flatten(net)
        net = slim.fully_connected(net, 192)
        net = slim.fully_connected(net, num_classes, activation_fn=None)
        return net

#load batch data
def load_batch(dataset, batch_size=5, height=299, width=299, is_training=False):
    """Loads a single batch of data.

        Args:
          dataset: The dataset to load.
          batch_size: The number of images in the batch.
          height: The size of each image after preprocessing.
          width: The size of each image after preprocessing.
          is_training: Whether or not we're currently training or evaluating.

        Returns:
          images: A Tensor of size [batch_size, height, width, 3], image samples that have been preprocessed.
          images_raw: A Tensor of size [batch_size, height, width, 3], image samples that can be used for visualization.
          labels: A Tensor of size [batch_size], whose values range between 0 and dataset.num_classes.
        """
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
             dataset, common_queue_capacity=32, common_queue_min=1)
    image_raw,  label = data_provider.get(['image', 'label'])

    image = inception_preprocessing.preprocess_image(image_raw, height, width, is_training=is_training)

    image_raw = tf.expand_dims(image_raw, 0)
    image_raw = tf.image.resize_images(image_raw, [height, width])
    image_raw = tf.squeeze(image_raw)

    images, image_raw, labels = tf.train.batch([image, image_raw, label],
                                               batch_size=batch_size,
                                               num_threads=1,
                                               capacity=2 * batch_size)
    return images, image_raw, labels

train_dir='/tmp/flowers'
print('will save model to %s' %train_dir)
with tf.Graph().as_default():
   tf.logging.set_verbosity(tf.logging.INFO)
   dataset = flowers.get_split('train', flowers_data_dir)
   images, _, labels = load_batch(dataset)

   logits = my_cnn(images, num_classes=dataset.num_classes, is_training=True)

   predictions = tf.argmax(logits,  1)
   names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'eval/Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
        'eval/Recall@3': slim.metrics.streaming_recall_at_k(logits, labels, 3)
   })

   print("running evaluation loop")
   checkpoint_path = tf.train.latest_checkpoint(train_dir)
   metrics_values = slim.evaluation.evaluate_once(
       master='',
       checkpoint_path=checkpoint_path,
       logdir=train_dir,
       eval_op=names_to_updates.values(),
       final_op=names_to_values.values()
   )

   names_to_values = dict(zip(names_to_values.keys(), metrics_values))

   for name in names_to_values:
       print ('%s: %f' %(name, names_to_values[name]))