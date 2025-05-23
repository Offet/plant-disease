import streamlit as st
import tensorflow as tf
import numpy as np

# tensorflow model prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.keras")
    # prep
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
    # convert to array
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    # convert to single img to batch
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index


# adding a sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select page", ["Home", "About", "Predict Disease"])

# separately define the sidebar menu
if(app_mode=="Home"):
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpg"
    st.image(image_path, use_container_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. 
    This tool is designed for farmers, gardeners, and plant enthusiasts to ensure healthy plants and crops.

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!
                
    The prediction can be done for any of these crops: Apple, Blueberry, Cherry (including sour), Corn (maize), Grape, Orange, Peach, Pepper, bell, Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato.        
    Together, let's protect our crops and ensure a healthier harvest!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)


    # do forr about page
elif(app_mode=="About"):
    st.header("About")
    image_path = "about.jpg"
    st.image(image_path, use_container_width=True)
    st.markdown("""
    #### About Our Data
    Our model was built on a dataset which consists of about 87,000 images of healthy and diseased crop leaves which is categorized different classes.The total data is split into 80/20 ratio of training and validation set preserving the directory structure.
    A new directory containing 33 test images is created later for prediction purpose.
    #### Content
    1. The training set contains 70295 images
    2. Test set contains 33 images
    3. Validation set has 17572 images

                """)
    


    # Prediction page
elif(app_mode=="Predict Disease"):
    st.header("Predict Disease")
    image_path = "prediction.jpg"
    st.image(image_path, use_container_width=True)
    test_image = st.file_uploader("Choose an image:")
    if(st.button("Show Image")):
        st.image(test_image, use_container_width=True)
    # prediction
    if(st.button("Predict")):
        with st.spinner("Please wait..."):
            st.write("Prediction")
            result_index = model_prediction(test_image)
            # define our classnames
            class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                        'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                        'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                        'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                        'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                        'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                        'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                        'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                        'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                        'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                        'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                        'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                        'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                        'Tomato___healthy']
            st.success("This is an image of {}".format(class_name[result_index]))
