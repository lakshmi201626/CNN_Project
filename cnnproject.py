import streamlit as st
from PIL import Image
from keras.models import load_model
import numpy as np
import cv2

def main():
    st.title('Car image prediction')
    st.write("This project develops a CNN-powered system for classifying car images."
             "The model leverages a simple CNN structure with five alternating convolutional and pooling layers."
             "While the high training accuracy (97%) is promising, the lower testing accuracy (76%) indicates the "
             "model's ability to generalize to new data might be limited")
    uploaded_file=st.file_uploader('upload your file')

    if uploaded_file is not None:
        prediction = st.button('Prediction')
        model = load_model('cnnmodel.h5')

        confidence_threshold = 0.8

        if prediction:
            # st.write(uploaded_file)
            image_bytes = uploaded_file.read()
            newimg = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)  # Assuming color image

            gray_img = cv2.cvtColor(newimg, cv2.COLOR_BGR2GRAY)
            img_resized = cv2.resize(gray_img, (150, 150))
            img_pred = img_resized.reshape(1, 150, 150, 1)
            pred = model.predict(img_pred)

            Categories = ['Swift', 'Audi', 'Rolls Royce', 'Tata Safari', 'Toyota Innova']
            ind = pred.argmax(axis=1)
            predicted_class = Categories[ind.item()]
            confidence = pred.max()  # Assuming highest value represents confidence
            if confidence > confidence_threshold:
                st.success(f"Predicted Class: {predicted_class} (Confidence: {confidence:.2f})")
            else:
                st.warning(f"Prediction uncertain, consider uploading a different image. Predicted Class: {predicted_class} (Confidence: {confidence:.2f})")
    else:
        # User didn't upload an image
        st.success("Please upload an image before making a prediction.")

main()
