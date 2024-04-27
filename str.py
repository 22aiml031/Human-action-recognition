import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Load the model
model = load_model('keras_model.h5')

# Define the class names
class_names = ["sitting","using_laptop","hugging","sleeping","drinking",
               "clapping","dancing","cycling","calling","laughing",
               "eating","fighting","listening_to_music","running","texting"]

def predict(image):
    input_img = np.asarray(image.resize((224, 224)))  # resize to 224x224
    input_img = input_img.astype('float32') / 255.0  # normalize to [0, 1]
    input_img = np.expand_dims(input_img, axis=0)  # add batch dimension

    result = model.predict(input_img)
    prediction = np.argmax(result)  # find the class with the highest probability

    # Print probabilities for each class
    for i, prob in enumerate(result[0]):
        print(f"{class_names[i]}: {prob*100}%")

    return class_names[prediction]  # return the name of the predicted class

# Streamlit app
st.title('Human Activity Recognition Model')
st.write("This is a simple image classification web app to predict human activities ,Upload an image and the model will predict the activity shown in the image. The classes are: sitting, using laptop, hugging, sleeping, drinking, clapping, dancing, cycling, calling, laughing, eating, fighting, listening to music, running, and texting")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = predict(image)
    st.write('%s' % label)
