
import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import cv2
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load the pre-trained model
MODEL_PATH = "/content/drone_bird_detector.h5"  # Path to the saved model
model = tf.keras.models.load_model(MODEL_PATH)

# Configuration for uploaded images and results
OUTPUT_FOLDER = "/content/drive/MyDrive/Drone&Bird/BirdVsDrone/Output Image"  # Folder to save predictions
IMG_SIZE = 224                  # Image size for resizing

# Ensure the output folder exists
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Function to make predictions on an uploaded image
def predict_image(image_path):
    # Load and preprocess the image
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)

    # Prediction
    prediction = model.predict(img_input)[0][0]
    label = "Drone" if prediction > 0.5 else "Bird"

    # Add label to the image
    labeled_img = cv2.putText(
        img_resized.copy(),
        label,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    # Save the labeled image in the output folder
    output_path = os.path.join(OUTPUT_FOLDER, os.path.basename(image_path))
    cv2.imwrite(output_path, labeled_img)

    return label, output_path

# Streamlit UI Setup
st.set_page_config(page_title="Bird and Drone Detection", layout="centered")

# App Title
st.title("Bird and Drone Detection System 🦅 🚁")

# Upload Section
st.header("Upload an Image")
uploaded_file = st.file_uploader(
    "Choose an image file", type=["jpg", "jpeg", "png"]
)

# Display Image and Prediction
if uploaded_file is not None:
    # Save the uploaded image
    image_path = os.path.join("uploaded_images", uploaded_file.name)
    if not os.path.exists("uploaded_images"):
        os.makedirs("uploaded_images")
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Show the uploaded image
    st.image(Image.open(image_path), caption="Uploaded Image", use_column_width=True)

    # Predict the class of the uploaded image
    st.subheader("Prediction")
    with st.spinner("Classifying..."):
        label, output_path = predict_image(image_path)

    # Display prediction result
    st.success(f"The uploaded image is classified as: **{label}**")
    st.image(output_path, caption="Labeled Image", use_column_width=True)

    # Provide download option for the labeled image
    with open(output_path, "rb") as file:
        st.download_button(
            label="Download Labeled Image",
            data=file,
            file_name=os.path.basename(output_path),
            mime="image/jpeg",
        )

# Batch Testing Section
st.header("Batch Testing (Folder of Images)")
batch_folder = st.text_input(
    "Enter the path of a folder containing test images:"
)

if st.button("Run Batch Testing"):
    if os.path.exists(batch_folder):
        # Predict on all images in the batch folder
        y_true, y_pred = [], []
        st.write("Processing images...")

        for img_name in os.listdir(batch_folder):
            img_path = os.path.join(batch_folder, img_name)
            if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                # Predict and save labeled image
                label, output_path = predict_image(img_path)

                # True label determination from filename
                true_label = 1 if "drone" in img_name.lower() else 0
                y_true.append(true_label)
                y_pred.append(1 if label == "Drone" else 0)

        # Generate and Display Confusion Matrix
        st.subheader("Confusion Matrix")
        conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
        disp = ConfusionMatrixDisplay(
            confusion_matrix=conf_matrix,
            display_labels=["Bird", "Drone"],
        )
        fig, ax = plt.subplots()
        disp.plot(ax=ax, cmap=plt.cm.Blues)
        st.pyplot(fig)

        st.success("Batch testing completed! Labeled images are saved in the output folder.")
    else:
        st.error("The specified folder does not exist. Please check the path.")

# Footer
st.markdown("---")
st.markdown("**Developed by [Rohan Chandrakar]** | Powered by TensorFlow & Streamlit")
MODEL_PATH = "app.h5"  # Path to save the model
model.save(MODEL_PATH)
print(f"Model saved at: {MODEL_PATH}")
