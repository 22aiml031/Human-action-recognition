import streamlit as st
import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image 
from tensorflow.keras.models import model_from_json
import json
# Load the model architecture from the JSON file
with open('model.json', 'r') as f:
    model = model_from_json(f.read())

# Load the weights into the model
model.load_weights('weights.bins')

def test_predict(image):
    situation=["sitting","using_laptop","hugging","sleeping","drinking",
           "clapping","dancing","cycling","calling","laughing"
          ,"eating","fighting","listening_to_music","running","texting"]
    input_img = np.asarray(image.resize((160,160)))
    result = model.predict(np.asarray([input_img]))
#     print('result:{}'.format(result))
    itemindex = np.where(result==np.max(result))
    print('itemindex:{}'.format(itemindex))
    prediction = itemindex[1][0]
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

        # Process the images
        processed_image = test_predict(image)
        st.write(processed_image)

        # Display the output
        st.write("Image processed successfully!")

if __name__ == "__main__":
    main()