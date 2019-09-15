# GenerateHandwrittenDigits
Generate handwritten digits using a deep convolutional generative adversarial network (DCGAN) all while being GPU accelerated.
Credit to https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/contrib/eager/python/examples/generative_examples/dcgan.ipynb

Instead of gathering and classifying numbers from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/), this program generates
handwritten numbers by discriminating our outputs after every generation. As training progresses, the generator gets better at generating
images until the discriminator can no longer spot a fake image.
