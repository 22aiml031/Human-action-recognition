import streamlit as st
import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
model = load_model('model.h5')

# Function to predict the action
def test_predict(image):
    situation=['calling', 'clapping', 'cycling', 'dancing', 'drinking', 'eating', 'fighting',
                    'hugging', 'laughing', 'listening_to_music', 'running', 'sitting', 'sleeping',
                    'texting', 'using_laptop']
    input_img = np.asarray(image.resize((224,224)))
    input_img = input_img/255.0
    # preprocess_fn = tf.keras.applications.resnet.preprocess_input
    # input_img = preprocess_fn(input_img)
    result = model.predict(np.asarray([input_img]))
#     print('result:{}'.format(result))
    itemindex = np.argmax(result, axis=1)
    print('itemindex:{}'.format(itemindex))
    prediction = itemindex[0]
    print("probability: "+str(np.max(result)*100) + "%\nPredicted class : ", situation[prediction])
    return "probability: "+str(np.max(result)*100) + "%\nPredicted class : ", situation[prediction]


def main():
    st.title("Human action  Classifier")
    st.write("This is a simple image classification web app to predict human actions")
    st.write("Upload a human image and the model will predict the action")
    st.write("The model was trained on the following classes: sitting, using_laptop, hugging, sleeping, drinking, clapping, dancing, cycling, calling, laughing, eating, fighting, listening_to_music, running, texting")


    # Upload image
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Process the image
        processed_image = test_predict(image)
        st.write(processed_image)

        # Display the output
        st.write("Image processed successfully!")

if __name__ == "__main__":
    main()