import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from skimage.color import rgb2lab, lab2rgb, gray2rgb
import matplotlib.pyplot as plt
from tqdm import tqdm

import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, UpSampling2D, BatchNormalization, Activation
from keras.layers import Concatenate, MaxPooling2D, Dropout

# ============================================
# CONFIGURATION
# ============================================

IMAGE_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 10

# Use absolute paths relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(SCRIPT_DIR, "data", "Flowers")
GRAY_PATH = os.path.join(BASE_PATH, "flowers_grey")
COLOR_PATH = os.path.join(BASE_PATH, "flowers_colour")

# ============================================
# LOAD & PREPROCESS DATA
# ============================================

def load_data(image_size=IMAGE_SIZE):
    if not os.path.exists(GRAY_PATH):
        raise FileNotFoundError(f"GRAY_PATH does not exist: {GRAY_PATH}")
    if not os.path.exists(COLOR_PATH):
        raise FileNotFoundError(f"COLOR_PATH does not exist: {COLOR_PATH}")

    print(f"Loading from:\n  Grayscale: {GRAY_PATH}\n  Color: {COLOR_PATH}")

    X, Y = [], []
    filenames = sorted([f for f in os.listdir(GRAY_PATH) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    for file in tqdm(filenames, desc="Loading and processing images"):
        gray_img = Image.open(os.path.join(GRAY_PATH, file)).convert('L').resize((image_size, image_size))
        color_img = Image.open(os.path.join(COLOR_PATH, file)).convert('RGB').resize((image_size, image_size))

        lab = rgb2lab(np.array(color_img))
        L = np.array(gray_img) / 100.0
        ab = lab[:, :, 1:] / 128.0

        X.append(L.reshape(image_size, image_size, 1))
        Y.append(ab)

    X = np.array(X)
    Y = np.array(Y)
    return train_test_split(X, Y, test_size=0.2, random_state=42)

# ============================================
# LOAD USER IMAGE
# ============================================

def load_user_image(image_path, image_size=IMAGE_SIZE):
    gray_img = Image.open(image_path).convert('L').resize((image_size, image_size))
    L = np.array(gray_img) / 100.0
    return L.reshape(1, image_size, image_size, 1)

# ============================================
# MODEL DEFINITION (U-Net-like)
# ============================================

def build_model():
    inputs = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))

    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Dropout(0.5)(conv4)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)

    up5 = UpSampling2D(size=(2, 2))(conv4)
    merge5 = Concatenate()([conv3, up5])
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(merge5)

    up6 = UpSampling2D(size=(2, 2))(conv5)
    merge6 = Concatenate()([conv2, up6])
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(merge6)

    up7 = UpSampling2D(size=(2, 2))(conv6)
    merge7 = Concatenate()([conv1, up7])
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(merge7)

    output = Conv2D(2, 1, activation='tanh')(conv7)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='mse')
    return model

# ============================================
# TRAINING
# ============================================

def train_model(model, X_train, Y_train, X_val, Y_val):
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS
    )
    return history

# ============================================
# PREDICTION AND VISUALIZATION
# ============================================

def predict_and_display(model, X_val, original_grayscale, num_images=5):
    predictions = model.predict(X_val[:num_images])

    for i in range(num_images):
        lab_image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3))
        lab_image[:, :, 0] = original_grayscale[i][:, :, 0] * 100
        lab_image[:, :, 1:] = predictions[i] * 128

        rgb = lab2rgb(lab_image)
        plt.imshow(rgb)
        plt.axis("off")
        plt.show()

# ============================================
# USER IMAGE COLORIZATION
# ============================================

def colorize_user_image(model, image_path):
    print(f"Colorizing user image: {image_path}")
    L_channel = load_user_image(image_path)
    ab_channels = model.predict(L_channel)

    lab_image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3))
    lab_image[:, :, 0] = L_channel[0, :, :, 0] * 100
    lab_image[:, :, 1:] = ab_channels[0] * 128

    rgb = lab2rgb(lab_image)
    plt.imshow(rgb)
    plt.axis("off")
    plt.show()

# ============================================
# MAIN
# ============================================

def main():
    print("Loading data...")
    X_train, X_val, Y_train, Y_val = load_data()

    print("Building model...")
    model = build_model()
    model.summary()

    print("Training model...")
    train_model(model, X_train, Y_train, X_val, Y_val)

    print("Displaying predictions...")
    predict_and_display(model, X_val, X_val)

    user_image_path = os.path.join(SCRIPT_DIR, "input_images", "my_grayscale_image.png")
    colorize_user_image(model, user_image_path)

if __name__ == "__main__":
    main()
