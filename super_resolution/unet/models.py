
# script to keep models

import tensorflow as tf

class Conv2D_BN(tf.keras.layers.Layer):
    '''2D Conv kernel with Batch Normalization.'''
    
    def __init__(self, filters, kernel_size, activation=None, padding='same', name=None, **kwargs):
        super().__init__()
        conv_name = None if name is None else name+'_conv'
        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=None, padding=padding, name=conv_name)
        bn_name = None if name is None else name+'_bn'
        self.bn = tf.keras.layers.BatchNormalization(axis=-1, name=bn_name)
        self.act = None if activation is None else tf.keras.layers.Activation(activation)
    
    def call(self, inputs, training):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        if self.act is not None:
            x = self.act(x)
        return x

def unet_model(in_shape, use_batchnorm=False, train_residual=False):

    inputs = tf.keras.layers.Input(shape=in_shape, name='input')
    x = inputs

    if use_batchnorm:
        conv_fn = lambda filters, name=None: Conv2D_BN(filters=filters, kernel_size=(3, 3), activation='relu', padding='same', name=name)
    else:
        conv_fn = lambda filters, name=None: tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), activation='relu', padding='same', name=name)
    deconv_fn = lambda filters, name=None: tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=(4,4), strides=(2,2), padding='same', name=name)
    
    # lvl 1
    x = conv_fn(filters=32, name='c11')(x)
    x = conv_fn(filters=32, name='c12')(x)
    x_lvl1 = x
    # lvl 2
    x = tf.keras.layers.MaxPool2D((2,2), name='p1')(x)
    x = conv_fn(filters=64, name='c21')(x)
    x = conv_fn(filters=64, name='c22')(x)
    x_lvl2 = x
    # lvl 3
    x = tf.keras.layers.MaxPool2D((2,2), name='p2')(x)
    x = conv_fn(filters=128, name='c31')(x)
    x = conv_fn(filters=128, name='c32')(x)
    x_lvl3 = x
    # lvl 4
    x = tf.keras.layers.MaxPool2D((2,2), name='p3')(x)
    x = conv_fn(filters=256, name='c41')(x)
    x = conv_fn(filters=256, name='c42')(x)
    x_lvl4 = x
    # lvl 5
    x = tf.keras.layers.MaxPool2D((2,2), name='p4')(x)
    x = conv_fn(filters=512, name='c51')(x)
    x = conv_fn(filters=512, name='c52')(x)
#     x_lvl5 = x
    # lvl 4
    x = deconv_fn(filters=512, name='d5')(x)
    x = tf.keras.layers.Concatenate(axis=-1, name='cat_lvl4')([x, x_lvl4])
    x = conv_fn(filters=256, name='dc41')(x)
    x = conv_fn(filters=256, name='dc42')(x)
    # lvl 3
    x = deconv_fn(filters=256, name='d4')(x)
    x = tf.keras.layers.Concatenate(axis=-1, name='cat_lvl3')([x, x_lvl3])
    x = conv_fn(filters=128, name='dc31')(x)
    x = conv_fn(filters=128, name='dc32')(x)
    # lvl 2
    x = deconv_fn(filters=128, name='d3')(x)
    x = tf.keras.layers.Concatenate(axis=-1, name='cat_lvl2')([x, x_lvl2])
    x = conv_fn(filters=64, name='dc21')(x)
    x = conv_fn(filters=64, name='dc22')(x)
    # lvl 1
    x = deconv_fn(filters=64, name='d2')(x)
    x = tf.keras.layers.Concatenate(axis=-1, name='cat_lvl1')([x, x_lvl1])
    x = conv_fn(filters=32, name='dc11')(x)
    x = conv_fn(filters=32, name='dc12')(x)
    # output lvl
    x = tf.keras.layers.Conv2D(filters=1, kernel_size=(1,1), activation=None, name='preds')(x)
#     x = tf.keras.layers.Conv2D(filters=1, kernel_size=(1,1), activation='sigmoid', name='preds')(x)
#     model = tf.keras.Model(inputs={'x': inputs}, outputs={'y': x}, name='unet')
    
    if train_residual:
        x = tf.keras.layers.Add(name='add_input')([x, inputs])
    
    model = tf.keras.Model(inputs=inputs, outputs=x, name='unet')

    return model


