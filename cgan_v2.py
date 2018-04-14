from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, MaxoutDense
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD

import matplotlib.pyplot as plt

import numpy as np

class CGAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = 10
        self.latent_dim = 100

        optimizer = SGD(lr=0.1, momentum=0.5, decay=1.00004)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss=['binary_crossentropy'],
            optimizer=optimizer)

        # print("metrics")
        # print(self.generator.metrics_names)

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(100,))
        label = Input(shape=(1,))
        img = self.generator([noise, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid = self.discriminator([img, label])

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = Model([noise, label], valid)
        self.combined.compile(loss=['binary_crossentropy'],
            optimizer=optimizer)

    def build_generator(self):

        #  Noise layer
        noise = Input(shape=(self.latent_dim,))
        noise_emb = Dense(200, activation='relu')(noise)      # out_shape: (batch_size, 200)
        noise_emb = Activation('relu')

        # Label layer
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))
        label_emb = Dense(1000, activation='relu')(label_embedding)     # out_shape: (batch_size, 1000)
        label_emb = Activation('relu')

        # Joint layers
        merged = Concatenate()([noise_emb, label_emb])        # out_shape: (batch_size, 1200)
        combined_out = Dense(1200, activation='relu')(merged) # out_shape: (batch_size, 1200)
        img_flat = Dense(np.prod(self.img_shape), activation='sigmoid') \
                                (combined_out)          # out_shape: (batch_size, 784)
        img = Reshape(self.img_shape)(img_flat)               # out_shape: (batch_size, 28, 28, 1)


        return Model([noise, label], img)    


    def build_discriminator(self):

        model = Sequential()

        img = Input(shape=self.img_shape)
        img_emb = MaxoutDense(240, nb_feature=5, input_dim=np.prod(self.img_shape))
        img_emb = Activation('relu')
        img_emb = Dropout(0.5)

        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, np.prod(self.img_shape))(label))
        label_emb_d = MaxoutDense(50, nb_feature=5)
        label_emb_d = Activation('relu')
        label_emb_d = Dropout(0.5)

        merged_d = Concatenate()([img_emb, label_emb_d]) 
        combined_d = MaxoutDense(240, nb_feature=4)
        combined_d = Activation('relu')
        combined_d = Dropout(0.5)

        img_d = Dense(1, activation='sigmoid')
        
        # model.summary()
        
        # flat_img = Flatten()(img_d)

        validity = combined_d(merged_d)

        return Model([img, label], validity)

    def train(self, epochs, batch_size=100, save_interval=50):

        # Load the dataset
        (X_train, y_train), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)
        y_train = y_train.reshape(-1, 1)


        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs, labels = X_train[idx], y_train[idx]

            noise = np.random.normal(0, 1, (batch_size, 100))

            gen_imgs = self.generator.predict([noise, labels])

            valid = np.ones((batch_size, 1))
            fake = np.zeros((batch_size, 1))

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid)
            # print(d_loss_real)
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
            # print(d_loss_fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            # print(d_loss)
            

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, 100))

            valid = np.ones((batch_size, 1))
            # Generator wants discriminator to label the generated images as the intended
            # digits
            sampled_labels = np.random.randint(0, 10, batch_size).reshape(-1, 1)

            # Train the generator
            g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)
            # print("g loss")
            # print(g_loss)
            # print("")

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 2, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        sampled_labels = np.arange(0, 10).reshape(-1, 1)

        gen_imgs = self.generator.predict([noise, sampled_labels])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        fig.suptitle("CGAN: Generated digits", fontsize=12)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt,:,:,0], cmap='gray')
                axs[i,j].set_title("Digit: %d" % sampled_labels[cnt])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    cgan = CGAN()
    cgan.train(epochs=2000, batch_size=32, save_interval=200)
