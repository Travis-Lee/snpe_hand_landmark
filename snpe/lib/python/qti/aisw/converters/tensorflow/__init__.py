import tensorflow as tf
try:
    tf_compat_v1 = tf.compat.v1
    tf.compat.v1.disable_eager_execution()
except AttributeError:
    tf_compat_v1 = tf

    # import contrib ops since they are not imported as part of TF by default
    import tensorflow.contrib
