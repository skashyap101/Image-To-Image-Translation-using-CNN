🌸 Deep Learning Image Colorization
This project uses a U-Net-like convolutional neural network to colorize grayscale images of flowers. It is built using TensorFlow/Keras, and trained on grayscale and color pairs of flower images.

📁 Folder Structure
![image](https://github.com/user-attachments/assets/a0b270e4-5ac1-450f-831f-d81a0b7bedc8)


📦 Requirements
Install dependencies with:

bash
Copy code
pip install -r requirements.txt
requirements.txt:

text
Copy code
numpy
matplotlib
Pillow
scikit-image
scikit-learn
tensorflow
tqdm
🚀 Running the Project
1. Prepare the Dataset
Place grayscale and color images in the following directories:

data/Flowers/flowers_grey/ — Grayscale flower images

data/Flowers/flowers_colour/ — Same images in RGB format (for training target)

Make sure each image in flowers_grey/ has an identically named image in flowers_colour/.

2. Run Training and Prediction
Run the main script:

bash
Copy code
python colorize.py
This will:

Train the model on the flower dataset

Display some colorized predictions from validation data

Attempt to colorize the user's grayscale image at input_images/my_grayscale_image.png

🖼️ Output
Colorized images are displayed using matplotlib. If you want to save them instead of just displaying, you can modify the colorize_user_image() function to use:

python
plt.imsave('output.png', rgb)
🧠 Model Architecture
The model is based on a U-Net-like architecture with skip connections, upsampling, and batch normalization.

📌 Notes
All images are resized to 256x256.

Colorization is done in Lab color space, predicting the a and b channels from a grayscale L channel.

Screenshot
![image](https://github.com/user-attachments/assets/2dffa789-db53-4839-a422-41a3b8975ca7)
![image](https://github.com/user-attachments/assets/4ed4cd7f-6b02-4e15-8ccc-d300897cbeae)

