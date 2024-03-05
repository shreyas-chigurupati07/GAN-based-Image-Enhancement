# Import Libraries
from imageio import imread
from PIL import Image
import os
import re
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from skimage.io import imread
from skimage.transform import resize
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, Conv2DTranspose, Concatenate, Dropout, UpSampling2D, Dense, Flatten, Reshape
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Constants
IMAGE_HEIGHT = 800
IMAGE_WIDTH = 800
BATCH_SIZE = 1
EPOCHS = 50

# Directories
REAL_SENSE_DIR = '/content/drive/MyDrive/benchmarking_project/dataset1/realsense_images'
ZED_DIR = '/content/drive/MyDrive/benchmarking_project/dataset1/zed2i'

generator_losses = []
discriminator_losses = []
discriminator_accuracies = []
psnr_values = []
ssim_values = []

# Utility functions


def preprocess_image(file_path, is_real_sense=True):
    image = imread(file_path, as_gray=is_real_sense).astype(np.float32)
    if is_real_sense:
        # Normalize RealSense images between -1 and 1 for tanh activation
        image = (image / 621.0) * 2 - 1
    else:
        # Normalize ZED images between -1 and 1 for tanh activation
        image = (image[:, :, 0] / 255.0) * 2 - 1
    image_resized = resize(
        image, (IMAGE_HEIGHT, IMAGE_WIDTH), anti_aliasing=True)
    image_resized = np.expand_dims(image_resized, axis=-1)
    return image_resized


# Preprocess and pair RealSense with ZED images
def preprocess_and_pair_images(real_sense_dir, zed_dir, IMAGE_HEIGHT, IMAGE_WIDTH):
    all_paired_images = []

    for object_folder in sorted(os.listdir(real_sense_dir)):
        real_sense_subdir = os.path.join(real_sense_dir, object_folder)
        zed_subdir = os.path.join(zed_dir, object_folder)

        if os.path.isdir(real_sense_subdir) and os.path.isdir(zed_subdir):
            files = sorted([f for f in os.listdir(
                real_sense_subdir) if f.endswith('.png')])

            for file_name in files:
                real_sense_path = os.path.join(real_sense_subdir, file_name)
                zed_path = os.path.join(zed_subdir, file_name)

                if os.path.exists(zed_path):
                    try:
                        real_sense_image = preprocess_image(
                            real_sense_path, is_real_sense=True)
                        zed_image = preprocess_image(
                            zed_path, is_real_sense=False)
                        all_paired_images.append((real_sense_image, zed_image))
                    except Exception as e:
                        print(
                            f"Error processing file pair {real_sense_path} and {zed_path}: {e}")
                else:
                    print(f"Matching ZED file not found for {file_name}")
        else:
            print(
                f"Subdirectory for object {object_folder} not found in both RealSense and ZED directories")

    print(f"Total paired images: {len(all_paired_images)}")
    return np.array(all_paired_images)


# Generator Model
def make_generator_model(input_shape):

    noisy_input = Input(shape=input_shape, name='noisy_input')

    zed_condition = Input(shape=input_shape, name='zed_condition')

    merged_input = Concatenate()([noisy_input, zed_condition])

    # Convolution layers
    conv1 = Conv2D(64, kernel_size=4, strides=2, padding='same')(merged_input)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU(alpha=0.2)(conv1)

    conv2 = Conv2D(128, kernel_size=4, strides=2, padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = LeakyReLU(alpha=0.2)(conv2)

    conv3 = Conv2D(256, kernel_size=4, strides=2, padding='same')(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = LeakyReLU(alpha=0.2)(conv3)
    conv3 = Dropout(0.5)(conv3)

    # Upsampling layers
    upsample1 = UpSampling2D()(conv3)
    upsample1 = Conv2D(512, kernel_size=4, padding='same')(upsample1)
    upsample1 = BatchNormalization()(upsample1)
    upsample1 = LeakyReLU(alpha=0.2)(upsample1)

    upsample2 = UpSampling2D()(upsample1)
    upsample2 = Conv2D(256, kernel_size=4, padding='same')(upsample2)
    upsample2 = BatchNormalization()(upsample2)
    upsample2 = LeakyReLU(alpha=0.2)(upsample2)

    upsample3 = UpSampling2D()(upsample2)
    upsample3 = Conv2D(128, kernel_size=4, padding='same')(upsample3)
    upsample3 = BatchNormalization()(upsample3)
    upsample3 = LeakyReLU(alpha=0.2)(upsample3)

    output_img = Conv2D(1, kernel_size=4, padding='same',
                        activation='tanh')(upsample3)

    model = Model(inputs=[noisy_input, zed_condition],
                  outputs=output_img, name='generator')

    return model

# Discriminator Model


def make_discriminator_model(input_shape):

    source_img = Input(shape=input_shape, name='source_input')
    target_img = Input(shape=input_shape, name='target_input')

    merged_input = Concatenate()([source_img, target_img])

    # Convolutional layers
    conv1 = Conv2D(64, kernel_size=4, strides=2, padding='same')(merged_input)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU(alpha=0.2)(conv1)

    conv2 = Conv2D(128, kernel_size=4, strides=2, padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = LeakyReLU(alpha=0.2)(conv2)

    conv3 = Conv2D(256, kernel_size=4, strides=2, padding='same')(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = LeakyReLU(alpha=0.2)(conv3)
    conv3 = Dropout(0.25)(conv3)

    flat = Flatten()(conv3)
    validity = Dense(1, activation='sigmoid')(
        flat)

    model = Model(inputs=[source_img, target_img],
                  outputs=validity, name='discriminator')

    return model


# Create the generator and discriminator models
image_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, 1)
generator = make_generator_model(image_shape)
discriminator = make_discriminator_model(image_shape)


# Compile the generator with Adam optimizer
generator_optimizer = Adam(2e-4, beta_1=0.5)
generator.compile(loss='binary_crossentropy', optimizer=generator_optimizer)

# Compile the discriminator with SGD optimizer
discriminator_optimizer = SGD(2e-4)
discriminator.compile(loss='binary_crossentropy',
                      optimizer=discriminator_optimizer, metrics=['accuracy'])

# Combined model
discriminator.trainable = False
real_sense_input = Input(shape=image_shape)
zed_condition_input = Input(shape=image_shape)
fake_img = generator([real_sense_input, zed_condition_input])
validity = discriminator([real_sense_input, fake_img])
combined_model = Model(
    inputs=[real_sense_input, zed_condition_input], outputs=validity)
combined_model.compile(loss='binary_crossentropy',
                       optimizer=generator_optimizer)

# Load and preprocess the dataset
paired_dataset = preprocess_and_pair_images(
    REAL_SENSE_DIR, ZED_DIR, IMAGE_HEIGHT, IMAGE_WIDTH)
source_images, target_images = paired_dataset[:, 0], paired_dataset[:, 1]

source_images_train, source_images_val, target_images_train, target_images_val = train_test_split(
    source_images, target_images, test_size=0.2, random_state=42)

# Define callbacks for checkpointing and early stopping
checkpoint_callback = ModelCheckpoint(
    'best_generator.h5', monitor='val_loss', save_best_only=True, mode='min')

early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=1,
    restore_best_weights=True)


def evaluate_model(generator, discriminator, real_sense_images, zed_images):
    d_accuracies = []
    g_losses = []
    psnr_values = []
    ssim_values = []
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    fake_zed_images = generator.predict([real_sense_images, zed_images])

    real_decision = discriminator.predict([real_sense_images, zed_images])
    fake_decision = discriminator.predict([real_sense_images, fake_zed_images])

    # Calculate metrics
    for real, fake in zip(real_decision, fake_decision):
        d_accuracies.append(np.mean(real > 0.5))
        g_losses.append(cross_entropy(np.ones_like(fake), fake).numpy())

    for real_img, fake_img in zip(zed_images, fake_zed_images):
        psnr_values.append(tf.image.psnr(
            real_img, fake_img, max_val=1.0).numpy())
        ssim_values.append(tf.image.ssim(
            real_img, fake_img, max_val=1.0).numpy())

    # Compute average metrics
    average_d_accuracy = np.mean(d_accuracies)
    average_g_loss = np.mean(g_losses)
    average_psnr = np.mean(psnr_values)
    average_ssim = np.mean(ssim_values)

    return average_d_accuracy, average_g_loss, average_psnr, average_ssim


# Plotting
def plot_overall_metrics(gen_loss, disc_loss, disc_acc, psnr_val, ssim_val):
    metrics = ['Generator Loss', 'Discriminator Loss',
               'Discriminator Accuracy', 'PSNR', 'SSIM']
    values = [gen_loss, disc_loss, disc_acc, psnr_val, ssim_val]
    colors = ['blue', 'red', 'green', 'orange', 'purple']

    plt.figure(figsize=(12, 8))

    for i, metric in enumerate(metrics):
        plt.subplot(2, 3, i + 1)
        plt.bar(metric, values[i], color=colors[i])
        plt.ylabel('Value')
        plt.title(metric)

    plt.tight_layout()
    plt.show()


# Training loop
for epoch in range(EPOCHS):
    epoch_gen_loss = []
    epoch_disc_loss = []
    epoch_disc_acc = []
    epoch_psnr = []
    epoch_ssim = []
    print(f"Epoch {epoch+1}/{EPOCHS}")
    for i in range(len(source_images_train)):

        real_sense_batch = source_images_train[i *
                                               BATCH_SIZE: (i + 1) * BATCH_SIZE]
        zed_batch = target_images_train[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]

        generated_images = generator.predict([real_sense_batch, zed_batch])

        valid = np.random.uniform(0.7, 1.2, (BATCH_SIZE, 1))
        fake = np.random.uniform(0.0, 0.3, (BATCH_SIZE, 1))

        if np.random.rand() < 0.1:
            valid, fake = fake, valid

        discriminator_loss_real = discriminator.train_on_batch(
            [real_sense_batch, zed_batch], valid)
        discriminator_loss_fake = discriminator.train_on_batch(
            [real_sense_batch, generated_images], fake)
        discriminator_loss = 0.5 * \
            np.add(discriminator_loss_real, discriminator_loss_fake)

        generator_loss = combined_model.train_on_batch(
            [real_sense_batch, zed_batch], valid)

        print(
            f"{i}/{len(source_images_train)} [D loss: {discriminator_loss[0]}, acc.: {100*discriminator_loss[1]}] [G loss: {generator_loss}]")

        epoch_gen_loss.append(generator_loss)
        epoch_disc_loss.append(discriminator_loss[0])
        epoch_disc_acc.append(discriminator_loss[1])

        # Store the metrics for each batch
        _, _, batch_psnr, batch_ssim = evaluate_model(
            generator, discriminator, real_sense_batch, zed_batch)

        epoch_psnr.append(batch_psnr)
        epoch_ssim.append(batch_ssim)

    # Compute average metrics for the epoch
    generator_losses.append(np.mean(epoch_gen_loss))
    discriminator_losses.append(np.mean(epoch_disc_loss))
    discriminator_accuracies.append(np.mean(epoch_disc_acc))
    psnr_values.append(np.mean(epoch_psnr))
    ssim_values.append(np.mean(epoch_ssim))

    # Evaluate on validation set
    val_loss = combined_model.evaluate(
        [source_images_val, target_images_val], np.ones(len(source_images_val)))
    checkpoint_callback.on_epoch_end(epoch, logs={'val_loss': val_loss})

print("Training complete.")


# Calculate overall averages for the entire training
overall_gen_loss = np.mean(generator_losses)
overall_disc_loss = np.mean(discriminator_losses)
overall_disc_acc = np.mean(discriminator_accuracies)
overall_psnr = np.mean(psnr_values)
overall_ssim = np.mean(ssim_values)

# Plot_overall_metrics
plot_overall_metrics(overall_gen_loss, overall_disc_loss,
                     overall_disc_acc, overall_psnr, overall_ssim)


# Inference function
def enhance_image(generator, real_sense_image_array, zed_image_array):

    enhanced_image = generator.predict([np.expand_dims(real_sense_image_array, axis=0),
                                        np.expand_dims(zed_image_array, axis=0)])

    return np.squeeze(enhanced_image, axis=0)


# Visualization function for Enhanced Image only
def plot_enhanced_image(enhanced_image):
    plt.figure(figsize=(6, 6))
    plt.title('Enhanced Image')
    plt.imshow(enhanced_image.squeeze(), cmap='gray')
    plt.axis('off')
    plt.show()


# Sample usage of the code:
sample_real_sense_path = '/content/drive/MyDrive/benchmarking_project/dataset1/test_images/realsense_images/medium_clamp/050.png'
sample_real_sense_image_preprocessed = preprocess_image(
    sample_real_sense_path, is_real_sense=True)

sample_zed_path = '/content/drive/MyDrive/benchmarking_project/dataset1/test_images/zed2i/medium_clamp/0100.png'
sample_zed_image_preprocessed = preprocess_image(
    sample_zed_path, is_real_sense=False)

enhanced_sample_image = enhance_image(
    generator, sample_real_sense_image_preprocessed, sample_zed_image_preprocessed)

plot_enhanced_image(enhanced_sample_image)
