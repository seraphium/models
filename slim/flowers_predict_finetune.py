import os
from nets import inception
from preprocessing import inception_preprocessing
from tensorflow.contrib import slim
import tensorflow as tf
from datasets import dataset_utils
from datasets import flowers
import numpy as np
from datasets import imagenet


url = "http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz"


image_size = inception.inception_v1.default_image_size
batch_size = 5

train_dir = '/tmp/inception_finetuned'
flowers_data_dir = "/home/jackie/dev/data/flowers"



#load batch data
def load_batch(dataset, batch_size=batch_size, height=299, width=299, is_training=False):
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


with tf.Graph().as_default():
    tf.logging.set_verbosity(tf.logging.INFO)
    dataset = flowers.get_split('train', flowers_data_dir)
    images, images_raw,  labels = load_batch(dataset, height=image_size, width=image_size)

    with slim.arg_scope(inception.inception_v1_arg_scope()):
        logits, _ = inception.inception_v1(images, num_classes=dataset.num_classes, is_training=True)

    probabilities = tf.nn.softmax(logits)
    checkpoint_path = tf.train.latest_checkpoint(train_dir)

    init_fn = slim.assign_from_checkpoint_fn(checkpoint_path, slim.get_variables_to_restore())

    with tf.Session() as sess:
        with slim.queues.QueueRunners(sess):
            sess.run(tf.initialize_local_variables())
            init_fn(sess)
            np_probabilities, np_image_raw, np_labels = sess.run([probabilities, images_raw, labels])
            for i in range(batch_size):
                image = np_image_raw[i, :, :, :,]
                true_label = np_labels[i]
                predicted_label = np.argmax(np_probabilities[i, :])
                predicted_name = dataset.labels_to_names[predicted_label]
                true_name = dataset.labels_to_names[true_label]
                print("%d: predict: %s  true:%s" % (i, predicted_name, true_name))


