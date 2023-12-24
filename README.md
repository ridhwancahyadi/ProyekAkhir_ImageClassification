# Rock, Paper, Scissors Classifier

This repository features a Rock, Paper, Scissors image classifier built using TensorFlow and Keras. This project was done to fulfill the Submission in Dicoding _Belajar Machine Learning untuk Pemula Course_

## About the Project

### Dataset

The project uses the Rock, Paper, Scissors dataset, which can be found [here](https://github.com/dicodingacademy/assets/releases/download/release/rockpaperscissors.zip). The dataset consists of images of hands showing rock, paper, or scissors gestures.

### Model Architecture

The image classifier model is built using Convolutional Neural Networks (CNN) with the following architecture:

```
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    # Flatten
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])
```

### How to Use

1. Clone the repository: git clone https://github.com/ridhwancahyadi/rock-paper-scissors-classifier.git
2. Open the test_rockpaper_scissors.ipynb notebook in a Jupyter environment.
3. Run the notebook cells to train the model and evaluate its performance.
4. After training, you can use the model to predict new images. The notebook includes an example of predicting rock, paper, or scissors from uploaded images.

### Prediction

The model can predict whether an image contains rock, paper, or scissors gestures. Try uploading your own images to test the model!

## Acknowledgments

Special thanks to Dicoding Academy for providing the dataset and opportunity to learn and create this project.
