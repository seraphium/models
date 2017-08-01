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

checkpoints_dir = '/tmp/inceptionv1'

# if not tf.gfile.Exists(checkpoints_dir):
#     tf.gfile.MakeDirs(checkpoints_dir)
#
# dataset_utils.download_and_uncompress_tarball(url, checkpoints_dir)

image_size = inception.inception_v1.default_image_size


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

def get_init_fn():
    """Returns a function run by the chief worker to warm-start the training."""
    checkpoint_exclude_scopes = ['InceptionV1/Logits', 'InceptionV1/AuxLogits']
    exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclude in exclusions:
            if var.op.name.startswith(exclude):
                excluded = True
        if not excluded:
            variables_to_restore.append(var)

    return slim.assign_from_checkpoint_fn(
        os.path.join(checkpoints_dir, 'inception_v1.ckpt'), variables_to_restore)


train_dir = '/tmp/inception_finetuned'
flowers_data_dir = "/home/jackie/dev/data/flowers"

with tf.Graph().as_default():
    tf.logging.set_verbosity(tf.logging.INFO)
    dataset = flowers.get_split('train', flowers_data_dir)
    images, _,  labels = load_batch(dataset, height=image_size, width=image_size)

    with slim.arg_scope(inception.inception_v1_arg_scope()):
        logits, _ = inception.inception_v1(images, num_classes=dataset.num_classes, is_training=True)

    one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)
    slim.losses.softmax_cross_entropy(logits, one_hot_labels)
    total_loss = slim.losses.get_total_loss()

    tf.summary.scalar('Losses/Total_Losses', total_loss)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op  = slim.learning.create_train_op(total_loss, optimizer)

    final_loss = slim.learning.train(
        train_op,
        logdir=train_dir,
        init_fn=get_init_fn(),
        number_of_steps=2
    )

print('Finished training. Last batch loss %f' % final_loss)
