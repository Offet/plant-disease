# # to learn fast api
# import tensorflow as tf
# import numpy as np
# import uvicorn
# from fastapi import FastAPI
# from fastapi.openapi.docs import (
#     get_redoc_html,
#     get_swagger_ui_html,
#     get_swagger_ui_oauth2_redirect_html,
# )


# # tensorflow model prediction
# def model_prediction(test_image):
#     model = tf.keras.models.load_model("trained_model.keras")
#     # prep
#     image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
#     # convert to array
#     input_arr = tf.keras.preprocessing.image.img_to_array(image)
#     # convert to single img to batch
#     input_arr = np.array([input_arr])
#     prediction = model.predict(input_arr)
#     result_index = np.argmax(prediction)
#     return result_index

# # initialis app
# app = FastAPI(docs_url=None)

# @app.get("/docs", include_in_schema=False)
# async def custom_swagger_ui_html():
#     return get_swagger_ui_html(
#         openapi_url=app.openapi_url,
#         title=app.title + " - Swagger UI",
#         oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
#         swagger_js_url="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js",
#         swagger_css_url="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css",
#     )

# # route
# @app.get('/')
# async def index():
#     return{'text': 'Hello Tester'}

# @app.get('/items/{name}')
# async def get_item(name):
#     return{'text': name}

# # ml
# app.get("/predict/{name}")
# async def predict(name):
#     image_model = model.transform([name]).toarray()
#      # prep
#     image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
#     # convert to array
#     input_arr = tf.keras.preprocessing.image.img_to_array(image)
#     # convert to single img to batch
#     input_arr = np.array([input_arr])
#     prediction = model.predict(input_arr)
#     result_index = np.argmax(prediction)
#     return result_index



# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8000)


import tensorflow as tf
import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from PIL import Image
import io

# load tf model
try:
    model = tf.keras.models.load_model("trained_model.keras")
    # getting class names
    CLASS_NAMES = class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
    print("TensorFlow model loaded successfully.")
except Exception as e:
    print(f"Error loading TensorFlow model: {e}")
    model = None 

# initialising app
app = FastAPI(
    title="Plant Disease Prediction API",
    description="An API to predict plant diseases using a TensorFlow model.",
    docs_url=None, 
    redoc_url=None 
)

# creating my custom Swagger UI and ReDoc endpoints
from fastapi.openapi.docs import get_swagger_ui_html, get_swagger_ui_oauth2_redirect_html

@app.get("/docs", include_in_schema=False, response_class=HTMLResponse)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css",
    )

@app.get("/redoc", include_in_schema=False, response_class=HTMLResponse)
async def redoc_html():
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=app.title + " - ReDoc",
        redoc_js_url="https://unpkg.com/redoc@2.0.0-rc.55/bundles/redoc.standalone.js",
    )

# defining my routes

@app.get('/')
async def root():
    """
    Root endpoint for the API.
    """
    return {'message': 'Welcome to the Plant Disease Prediction API! Visit /docs for more information.'}

@app.get('/hello/{name}')
async def get_hello_name(name: str):
    """
    A simple endpoint to greet a name.
    """
    return {'message': f'Hello {name}'}

@app.post("/predict_image/")
async def predict_image(file: UploadFile = File(...)):
    """
    Upload an image file and get a plant disease prediction.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please check server logs.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

    try:
        # Read image content
        contents = await file.read()
        # Open image using PIL
        image = Image.open(io.BytesIO(contents))
        # Resize image to target_size (128, 128)
        image = image.resize((128, 128))

        # Convert PIL Image to TensorFlow array
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        # Expand dimensions to create a batch of 1 image
        input_arr = np.array([input_arr])

        # Make prediction
        prediction = model.predict(input_arr)
        result_index = np.argmax(prediction)

        # Get class name (assuming CLASS_NAMES is defined globally)
        predicted_class_name = CLASS_NAMES[result_index] if CLASS_NAMES and 0 <= result_index < len(CLASS_NAMES) else f"Class Index: {result_index}"

        return {
            "filename": file.filename,
            "prediction_index": int(result_index), 
            "predicted_class": predicted_class_name,
            "confidence": float(np.max(prediction)) 
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image or making prediction: {e}")

# --- Main execution block for Uvicorn ---
if __name__ == "__main__":
    # Ensure Uvicorn runs the correct app instance
    uvicorn.run(app, host="0.0.0.0", port=8000) 