import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Dropout, Input, TimeDistributed
from keras.optimizers import Adam
from keras import backend as K

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic time series data for training
def generate_real_data(n_samples, n_steps, n_features):
    X = np.random.rand(n_samples, n_steps, n_features)
    return X

# Define the generator model
def build_generator(latent_dim, n_steps, n_features):
    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, latent_dim)))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(n_features))
    return model

# Define the discriminator model
def build_discriminator(n_steps, n_features):
    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(LSTM(50, activation='relu'))
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
latent_dim = 5
n_steps = 10
n_features = 1
batch_size = 32
epochs = 5000

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
    X_real = generate_real_data(batch_size, n_steps, n_features)
    y_real = np.ones((batch_size, 1))
    
    # Generate fake data
    X_fake = generator.predict(np.random.randn(batch_size, n_steps, latent_dim))
    y_fake = np.zeros((batch_size, 1))
    
    # Train the discriminator
    d_loss_real = discriminator.train_on_batch(X_real, y_real)
    d_loss_fake = discriminator.train_on_batch(X_fake, y_fake)
    
    # Train the generator (via the GAN model, with flipped labels)
    y_gan = np.ones((batch_size, 1))
    g_loss = gan.train_on_batch(np.random.randn(batch_size, n_steps, latent_dim), y_gan)
    
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}/{epochs} - D Loss Real: {d_loss_real[0]}, D Loss Fake: {d_loss_fake[0]}, G Loss: {g_loss}')

# Generate and plot synthetic data
n_samples = 10
X_synthetic = generator.predict(np.random.randn(n_samples, n_steps, latent_dim))
for i in range(n_samples):
    plt.plot(X_synthetic[i], label=f'Sample {i+1}')
plt.title('Generated Time Series Data')
plt.legend()
plt.show()