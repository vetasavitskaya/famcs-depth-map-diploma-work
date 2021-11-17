import tensorflow.keras.backend as keras_b
import tensorflow as tf


def depth_loss_calculation(y_true, y_predicted, theta=0.1, maxDepthVal=1000.0 / 10.0):
    l_depth = keras_b.mean(keras_b.abs(y_predicted - y_true), axis=-1)

    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_predicted, dx_predicted = tf.image.image_gradients(y_predicted)

    l_grad = keras_b.mean(keras_b.abs(dx_predicted - dx_true) + keras_b.abs(dy_predicted - dy_true), axis=-1)

    l_ssim = keras_b.clip((1 - tf.image.ssim(y_true, y_predicted, maxDepthVal)) * 0.5, 0, 1)

    w1, w2, w3 = 1.0, 1.0, theta
    return (w1 * l_ssim) + (w2 * keras_b.mean(l_grad)) + (w3 * keras_b.mean(l_depth))
