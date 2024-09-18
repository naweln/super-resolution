
# Fully Dense Unet Training

import os
import sys
import time
import numpy as np
import pickle
import h5py
import tensorflow as tf

import generators
import models

tf.keras.backend.clear_session()
physical_devices = tf.config.experimental.list_physical_devices('GPU') 
for gpu_instance in physical_devices: 
    tf.config.experimental.set_memory_growth(gpu_instance, True)


def setup_datasets(fname_h5, batch_size, num_prefetch):
    
    x_dtype, y_dtype    = tf.float32, tf.float32

    # training dataset
    gen_train = generators.Generator_Simple(fname_h5=fname_h5)
    dataset_train = tf.data.Dataset.from_generator(lambda: gen_train,
                                                    output_shapes=gen_train.shapes,
                                                    output_types=(x_dtype, y_dtype))

    dataset_train = dataset_train.batch(batch_size, drop_remainder=True)
    dataset_train = dataset_train.prefetch(buffer_size=num_prefetch)

    # validation dataset
    gen_val = generators.Generator_Simple(fname_h5=fname_h5)
    dataset_val = tf.data.Dataset.from_generator(lambda: gen_val,
                                                    output_shapes=gen_val.shapes,
                                                    output_types=(x_dtype, y_dtype))

    dataset_val = dataset_val.batch(batch_size, drop_remainder=True)
    dataset_val = dataset_val.prefetch(buffer_size=num_prefetch)

    # test dataset
    gen_test = generators.Generator_Simple(fname_h5=fname_h5)
    dataset_test = tf.data.Dataset.from_generator(lambda: gen_test,
                                                    output_shapes=gen_test.shapes,
                                                    output_types=(x_dtype, y_dtype))
    
    dataset_test = dataset_test.batch(batch_size, drop_remainder=True)
    dataset_test = dataset_test.prefetch(buffer_size=num_prefetch)

    return dataset_train, dataset_val, dataset_test

# training step
@tf.function
def train_step(model, x, y, optimizer, loss_fn):
    '''Single training step'''
    with tf.GradientTape(persistent=False) as tape:
        y_pred      = model(x, training=True)
        loss        = loss_fn(y_true=y, y_pred=y_pred)
        
        gradients   = tape.gradient(target=loss, sources=model.trainable_variables)
        
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss, y_pred

#===== training parameters =====#

# defined in the order used below
numEpoch            = 20
batch_size          = 2
num_prefetch        = 5
imsize              = [256,256]

in_channels        = ['recon_linear']
out_channels       = ['recon_multisegment']

fname_h5           = 'data/parsed_armBP.h5'
dataset_train, dataset_val, dataset_test = setup_datasets(fname_h5, batch_size, num_prefetch)

# define model, optimizer and loss
model               = models.unet_model(in_shape=imsize + [1], use_batchnorm=True, train_residual=True)
lr_schedule         = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1.e-3, decay_steps=int(1e3), decay_rate=0.98, staircase=True)
optimizer           = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
loss_fn             = tf.keras.losses.MeanSquaredError()

# paths for logs
model_name_str      ='unet'
logPath             ='logs/'
logdir              = os.path.join(logPath, model_name_str)

#===== training iterations =====#
model.summary()

avg_loss_epoch      = tf.keras.metrics.Mean(name='loss', dtype=tf.float32)
avg_loss_log        = tf.keras.metrics.Mean(name='loss', dtype=tf.float32)

#===== training loop =====#
for epoch in range(0, numEpoch):
    for inputImage, targetImage in dataset_train:

        loss, predImage     = train_step(model=model, x=inputImage, y=targetImage, optimizer=optimizer, loss_fn=loss_fn)
        num_step            = optimizer.iterations

        # update average losses
        avg_loss_epoch.update_state(loss)

        # print step number and loss
        print('Step: %d Loss: %.3f' %(num_step, loss))

        # validation
        if tf.equal(num_step % 1000, 0):
            avg_loss_val = tf.keras.metrics.Mean(name='loss_', dtype=tf.float32)
            for x, y in dataset_val:
                y_pred      = model(x, training=False)
                eval_loss   = loss_fn(y_true=y, y_pred=y_pred)
                avg_loss_val.update_state(eval_loss)
            print('Step: %d Loss: %.3f' %(num_step, avg_loss_val))

    print('***Epoch %d***' %(epoch))
    print('Step: %d' %(num_step))
    print('Loss: %.3f' %(avg_loss_epoch.result()))
    print('**************')

    avg_loss_epoch.reset_states()

    ckpt            = tf.train.Checkpoint(optimizer=optimizer, model=model)
    save_manager    = tf.train.CheckpointManager(checkpoint=ckpt, directory=logdir, checkpoint_name='model_ckpt')
    save_manager.save(checkpoint_number=epoch)

# test loop
for x, y in dataset_test:
    avg_loss_test = tf.keras.metrics.Mean(name='loss_', dtype=tf.float32)
    for x, y in dataset_test:
        y_pred      = model(x, training=False)

        # print here test input and output if you want to see predictions

        test_loss   = loss_fn(y_true=y, y_pred=y_pred)
        avg_loss_test.update_state(eval_loss)
    print('Step: %d Loss: %.3f' %(num_step, avg_loss_test))


