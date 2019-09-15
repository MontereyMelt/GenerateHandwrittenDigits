# FILE: dcgan.py
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# DATE: 09/14/19  ||   PROGRAMMERS(S): Miguel Martinez
# ======================================================================================================================
# CREDITS:
# https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/contrib/eager/
#       python/examples/generative_examples/dcgan.ipynb (Thank you for the great tutorial)
# ======================================================================================================================
# PURPOSE: This Tensorflow tutorial demonstrates how to generate images of handwritten digits using a Deep Convolutional
# Generative Adversarial Network (DCGAN). The code is written in tf.keras with eager execution enabled.
# Tensorflow-gpu allows us to GPU accelerate our program and reduce the amount of time per epoch...
#                                                                                           Thank you cuda cores.
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# ----------------------------------------------------------------------------------------------------------------------
# Packages: Tensorflow-gpu 1.14 | imageio 2.5.0 | matplotlib.pyplot 3.1.1 | cudatoolkit 10.0.130 | cudnn 7.6.0
# ----------------------------------------------------------------------------------------------------------------------
import tensorflow as tf
tf.enable_eager_execution()

import glob
import imageio
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import PIL
import time

import IPython
from IPython import display

# Enable GPU session here. gpu_memory_fraction denotes what fraction of our dedicated GPU memory we will use for
# this session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
sess = tf.Session(config=config)

# Load the dataset. Here we use the MNIST dataset to train the generator and the discriminator. Our generator will
# generate handwritten digits resembling the MNIST data.
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]
BUFFER_SIZE = 60000
BATCH_SIZE = 256

# Use tf.data to create batches and shuffle the dataset
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# ----------------------------------------------------------------------------------------------------------------------
# Create the models
# ----------------------------------------------------------------------------------------------------------------------

# The generator is responsible for creating convincing images that are good enough to fool the discriminator.
# The network architecture for the generator consists of Conv2DTranspose (Upsampling) layers. We start with a fully
# connected layer and upsample the image two times in order to reach the desired image size of 28x28x1. We increase the
# width and height, and reduce the depth as we move through the layers in the network. We use Leaky ReLU activation for
# each layer except for the last one where we use a tanh activation.
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

    model.add(tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(
        tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model

# The discriminator is responsible for distinguishing fake images from real images.
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))

    return model

generator = make_generator_model()
discriminator = make_discriminator_model()

# ----------------------------------------------------------------------------------------------------------------------
# Define the loss functions and the optimizer
# ----------------------------------------------------------------------------------------------------------------------

# The generator loss is a sigmoid cross entropy loss of the generated images and an array of ones, since the generator
# is trying to generate fake images that resemble the real images.
def generator_loss(generated_output):
    return tf.losses.sigmoid_cross_entropy(tf.ones_like(generated_output), generated_output)

# The Discrimator loss function takes two input: real images, and generated images.
def discriminator_loss(real_output, generated_output):
    # [1,1,...,1] with real output since it is true and we want our generated examples to look like it
    real_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(real_output),
                                                logits=real_output)

    # [0,0,...,0] with generated images since they are fake
    generated_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.zeros_like(generated_output),
                                                     logits=generated_output)

    total_loss = real_loss + generated_loss

    return total_loss

generator_optimizer = tf.train.AdamOptimizer(1e-4)
discriminator_optimizer = tf.train.AdamOptimizer(1e-4)

# Checkpoints for saving our progress
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# ----------------------------------------------------------------------------------------------------------------------
# Setup GANs for training
# ----------------------------------------------------------------------------------------------------------------------

# Define training parameters
EPOCHS = 100
noise_dim = 100
num_examples_to_generate = 16
# We'll re-use this random vector used to seed the generator so it will be easier to see the improvement over time.
random_vector_for_generation = tf.random_normal([num_examples_to_generate,
                                                 noise_dim])

# Define training method.
# The generator is given a random vector as an input which is processed to output an image looking like a handwritten
# digit. The discriminator is then shown the real MNIST images as well as the generated images.
def train_step(images):
    # generating noise from a normal distribution
    noise = tf.random_normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        generated_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(generated_output)
        disc_loss = discriminator_loss(real_output, generated_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.variables))

# By using tf.contrib.eager.defun to create graph functions, we get a ~20 secs/epoch performance boost
# (from ~50 secs/epoch down to ~30 secs/epoch).
# On an Nvidia 1080ti utilizing 20% of dedicated memory, it is 8-9 secs per epoch. Using 90% only increased our
# sec/epoch speed by -0.5s (~7.5s)
train_step = tf.contrib.eager.defun(train_step)

# ----------------------------------------------------------------------------------------------------------------------
# Train the GANs
# ----------------------------------------------------------------------------------------------------------------------

# Note, training GANs can be tricky. It's important that the generator and discriminator do not overpower each other
# (e.g., that they train at a similar rate).
def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for images in dataset:
            train_step(images)

        display.clear_output(wait=True)
        generate_and_save_images(generator,
                                 epoch + 1,
                                 random_vector_for_generation)

        # saving (checkpoint) the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time taken for epoch {} is {} sec'.format(epoch + 1,
                                                         time.time() - start))
    # generating after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epochs,
                             random_vector_for_generation)

# Generate and save images
def generate_and_save_images(model, epoch, test_input):
    # make sure the training parameter is set to False because we
    # don't want to train the batchnorm layer when doing inference.
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    #plt.show()

# At the beginning of the training, the generated images look like random noise. As training progresses, you can see the
# generated digits look increasingly real. After 50 epochs, they look very much like the MNIST digits.
train(train_dataset, EPOCHS)

# Restore the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# ----------------------------------------------------------------------------------------------------------------------
# Generated images
# ----------------------------------------------------------------------------------------------------------------------

# Display a single image using the epoch number
def display_image(epoch_no):
  return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

display_image(EPOCHS)

# Generate a GIF of all the saved images
with imageio.get_writer('dcgan.gif', mode='I') as writer:
    filenames = glob.glob('image*.png')
    filenames = sorted(filenames)
    last = -1
    for i, filename in enumerate(filenames):
        frame = 2 * (i ** 0.5)
        if round(frame) > round(last):
            last = frame
        else:
            continue
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)