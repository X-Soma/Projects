import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# Load the Fashion MNIST dataset
fashion = keras.datasets.fashion_mnist
(xtrain, ytrain), (xtest, ytest) = fashion.load_data()

# Normalize the data
xtrain, xtest = xtrain / 255.0, xtest / 255.0

# Function to display an image
def display_image(image, label):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    ax.set_title(f"Label: {label}")
    buf = BytesIO()
    plt.savefig(buf, format="png")
    st.image(buf, use_column_width=True)
    plt.close()

# Sidebar: Select an image index
imgIndex = st.sidebar.slider("Select Image Index", 0, len(xtrain) - 1, 4)

# Display the selected image
st.write("### Selected Image from Training Dataset")
display_image(xtrain[imgIndex], ytrain[imgIndex])

# Display dataset shapes
st.write("### Dataset Shapes")
st.write(f"Training Data Shape: {xtrain.shape}")
st.write(f"Testing Data Shape: {xtest.shape}")

# Define the model
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

# Display model summary
st.write("### Model Summary")
with st.expander("Click to view model summary"):
    model_summary = []
    model.summary(print_fn=lambda x: model_summary.append(x))
    st.text("\n".join(model_summary))

# Train the model
if st.button("Train the Model"):
    xvalid, train = xtrain[:5000], xtrain[5000:]
    yvalid, ytrain = ytrain[:5000], ytrain[5000:]

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])

    st.write("Training in progress...")
    history = model.fit(train, ytrain, epochs=10, validation_data=(xvalid, yvalid), verbose=0)
    st.write("### Training Complete")

    # Plot training history
    st.write("### Training History")
    fig, ax = plt.subplots()
    ax.plot(history.history['accuracy'], label='Train Accuracy')
    ax.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.legend()
    st.pyplot(fig)

# Make predictions
if st.button("Predict on Test Images"):
    new = xtest[:5]
    predictions = model.predict(new)

    st.write("### Predictions")
    predicted_classes = np.argmax(predictions, axis=1)
    for i, pred in enumerate(predicted_classes):
        st.write(f"Image {i+1}: Predicted Label = {pred}, True Label = {ytest[i]}")
        display_image(new[i], ytest[i])
