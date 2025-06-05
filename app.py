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
    CLASS_NAMES = ['Apple - Apple scab', 'Apple - Black rot', 'Apple - Cedar apple rust', 'Apple - healthy', 'Blueberry - healthy', 'Cherry - Powdery mildew', 'Cherry - healthy', 'Corn (maize) - Cercospora leaf spot Gray leaf spot', 'Corn (maize) - Common rust ', 'Corn (maize) - Northern Leaf Blight', 'Corn (maize) - healthy', 'Grape - Black rot', 'Grape - Esca (Black Measles)', 'Grape - Leaf blight (Isariopsis Leaf Spot)', 'Grape - healthy', 'Orange - Haunglongbing (Citrus greening)', 'Peach - Bacterial spot', 'Peach - healthy', 'Pepper, bell - Bacterial spot', 'Pepper, bell - healthy', 'Potato - Early blight', 'Potato - Late blight', 'Potato - healthy', 'Raspberry - healthy', 'Soybean - healthy', 'Squash - Powdery mildew', 'Strawberry - Leaf scorch', 'Strawberry - healthy', 'Tomato - Bacterial spot', 'Tomato - Early blight', 'Tomato - Late blight', 'Tomato - Leaf Mold', 'Tomato - Septoria leaf spot', 'Tomato - Spider mites Two-spotted spider mite', 'Tomato - Target Spot', 'Tomato - Tomato Yellow Leaf Curl Virus', 'Tomato - Tomato mosaic virus', 'Tomato - healthy']
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


# endpoint to check if the API is running
@app.get('/')
async def root():
    """
    Root endpoint for the API.
    """
    return {"message": "Welcome to the Plant Disease Prediction API! Visit /docs for more information. \n\n"
                   "This API allows you to upload an image of a plant leaf and get a prediction of its health status. \n \n \n"
                   "The following crops are supported: Apple, Blueberry, Cherry (including sour), Corn (maize), Grape, Orange, Peach, Pepper (bell), Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato."}

# endpoint to greet a name
@app.get('/hello/{name}')
async def get_hello_name(name: str):
    """
    A simple endpoint to greet a name.
    """
    return {"message": f"Hello {name}"}

# endpoint to predict plant disease
@app.post("/predict/")
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

        # Preprocess the image for the model
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

         # Apply the confidence condition
        confidence = float(np.max(prediction)* 100)
        # if confidence < 80:
        #     return {
        #         "message": "Kindly take a clearer image of the plant leaf."
        #     }

        # Retrieve disease info
        disease_info = PLANT_DISEASE_INFO.get(predicted_class_name, {
            "description": "No detailed information available for this disease.",
            "solutions": ["Consult with a local agricultural expert for guidance."]
        })

        return {
            "predicted_class": predicted_class_name,
            "confidence": f"{confidence:.2f}%",
            "description": disease_info["description"],
            "solutions": disease_info["solutions"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image or making prediction: {e}")

# --- Main execution block for Uvicorn ---
if __name__ == "__main__":
    # Ensure Uvicorn runs the correct app instance
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port) 