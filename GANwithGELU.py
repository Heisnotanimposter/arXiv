import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Dropout, Input, TimeDistributed, Activation
from keras.optimizers import Adam
from keras import backend as K
from keras.activations import gelu
from keras.datasets import mnist
from keras.utils import to_categorical

# Set random seed for reproducibility
np.random.seed(42)

# Load and preprocess MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Define the generator model
def build_generator(latent_dim, n_steps, n_features):
    model = Sequential()
    model.add(LSTM(50, activation=gelu, return_sequences=True, input_shape=(n_steps, latent_dim)))
    model.add(LSTM(50, activation=gelu))
    model.add(Dense(n_features))
    return model

# Define the discriminator model
def build_discriminator(n_steps, n_features):
    model = Sequential()
    model.add(LSTM(50, activation=gelu, return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(LSTM(50, activation=gelu))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Define the GAN model
def build_gan(generator, discriminator):
    discriminator.trainable = False
    gan_input = Input(shape=(n_steps, latent_dim))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(gan_input, gan_output)
    return gan

# Set hyperparameters
latent_dim = 100
n_steps = 28
n_features = 28
batch_size = 64
epochs = 10000

# Build and compile the discriminator
discriminator = build_discriminator(n_steps, n_features)
discriminator.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Build the generator
generator = build_generator(latent_dim, n_steps, n_features)

# Build and compile the GAN
gan = build_gan(generator, discriminator)
gan.compile(optimizer=Adam(), loss='binary_crossentropy')

# Train the GAN
for epoch in range(epochs):
    # Generate real data
    idx = np.random.randint(0, x_train.shape[0], batch_size)
    X_real = x_train[idx]
    y_real = np.ones((batch_size, 1))
    
    # Generate fake data
    noise = np.random.randn(batch_size, n_steps, latent_dim)
    X_fake = generator.predict(noise)
    y_fake = np.zeros((batch_size, 1))
    
    # Train the discriminator
    d_loss_real = discriminator.train_on_batch(X_real, y_real)
    d_loss_fake = discriminator.train_on_batch(X_fake, y_fake)
    
    # Train the generator (via the GAN model, with flipped labels)
    y_gan = np.ones((batch_size, 1))
    g_loss = gan.train_on_batch(noise, y_gan)
    
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}/{epochs} - D Loss Real: {d_loss_real[0]}, D Loss Fake: {d_loss_fake[0]}, G Loss: {g_loss}')

# Generate and plot synthetic data
n_samples = 10
noise = np.random.randn(n_samples, n_steps, latent_dim)
X_synthetic = generator.predict(noise)

for i in range(n_samples):
    plt.imshow(X_synthetic[i].reshape((n_steps, n_features)), cmap='gray')
    plt.title(f'Synthetic Sample {i+1}')
    plt.show()