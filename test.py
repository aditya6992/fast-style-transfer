from src import transform
import numpy as np
import tensorflow as tf
import os
from src.utils import get_img

# g = tf.Graph()
# batch_size = 1
# curr_num = 0
# soft_config = tf.ConfigProto(allow_soft_placement=True)
# soft_config.gpu_options.allow_growth = True
# with g.as_default(), g.device("gpu:0"), \
#         tf.Session(config=soft_config) as sess:
#     batch_shape = (batch_size,) + img_shape
#     img_placeholder = tf.placeholder(tf.float32, shape=batch_shape,
#                                      name='img_placeholder')
#
#     preds = transform.net(img_placeholder)
#     saver = tf.train.Saver()
#     if os.path.isdir(checkpoint_dir):
#         ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
#         if ckpt and ckpt.model_checkpoint_path:
#             saver.restore(sess, ckpt.model_checkpoint_path)
#         else:
#             raise Exception("No checkpoint found...")
#     else:
#         saver.restore(sess, checkpoint_dir)
#
#     num_iters = int(len(paths_out)/batch_size)
#     for i in range(num_iters):
#         pos = i * batch_size
#         curr_batch_out = paths_out[pos:pos+batch_size]
#         if is_paths:
#             curr_batch_in = data_in[pos:pos+batch_size]
#             X = np.zeros(batch_shape, dtype=np.float32)
#             for j, path_in in enumerate(curr_batch_in):
#                 img = get_img(path_in)
#                 assert img.shape == img_shape, \
#                     'Images have different dimensions. ' +  \
#                     'Resize images or use --allow-different-dimensions.'
#                 X[j] = img
#         else:
#             X = data_in[pos:pos+batch_size]
#
#         _preds = sess.run(preds, feed_dict={img_placeholder:X})
#         for j, path_out in enumerate(curr_batch_out):
#             save_img(path_out, _preds[j])

def get_output(image_as_array, checkpoint_dir):
    g = tf.Graph()
    soft_config = tf.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True
    with g.as_default(), g.device("cpu:0"), tf.Session(config=soft_config) as sess:
        batch_shape = (1,) + image_as_array.shape
        batch = np.zeros(batch_shape, dtype=np.float32)
        batch[0,:,:,:] = image_as_array
        img_placeholder = tf.placeholder(tf.float32, shape=batch_shape,
                                         name='img_placeholder')
        preds = transform.net(img_placeholder)
        saver = tf.train.Saver()
        if os.path.isdir(checkpoint_dir):
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise Exception("No checkpoint found...")
        else:
            saver.restore(sess, checkpoint_dir)

        _preds = sess.run(preds, feed_dict={img_placeholder: batch})
        return _preds[0]

if __name__=="__main__":
    imagepath = "./examples/content/stata.jpg"
    checkpoint_dir = r"C:\Users\adity_000\Desktop\checkpoints"
    image = get_img(imagepath)

    import matplotlib.pyplot as plt

    res = get_output(image, checkpoint_dir)
    img = np.clip(res, 0, 255).astype(np.uint8)
    plt.imshow(img)
    plt.show()