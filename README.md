# Unimodal Conditional Generative Adversarial (CGAN) Network using Keras and MNIST dataset 

This is a Keras implementation of the priliminary model of the Unimodal CGAN presented in the paper -> https://arxiv.org/abs/1411.1784

The results of the code are not perfect but I tried to stick as much as possible to the network model mentioned in the paper. Quoting from the paper the network is as follows:

"In the generator net, a noise prior z with dimensionality 100 was drawn from a uniform distribution
within the unit hypercube. Both z and y are mapped to hidden layers with Rectified Linear Unit
(ReLu) activation, with layer sizes 200 and 1000 respectively, before both being mapped to
second, combined hidden ReLu layer of dimensionality 1200. We then have a final sigmoid unit
layer as our output for generating the 784-dimensional MNIST samples.

The discriminator maps x to a maxout layer with 240 units and 5 pieces, and y to a maxout layer
with 50 units and 5 pieces. Both of the hidden layers mapped to a joint maxout layer with 240 units
and 4 pieces before being fed to the sigmoid layer. (The precise architecture of the discriminator
is not critical as long as it has sufficient power; we have found that maxout units are typically well
suited to the task.)
The model was trained using stochastic gradient decent with mini-batches of size 100 and initial
learning rate of 0.1 which was exponentially decreased down to .000001 with decay factor of
1.00004. Also momentum was used with initial value of .5 which was increased up to 0.7. Dropout with probability of 0.5 was applied to both the generator and discriminator."


Original code base extrated from -> https://github.com/eriklindernoren/Keras-GAN

Update - cgan_v2.py file presents a more closer implementation to that of the paper
