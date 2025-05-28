# to learn fast api
import tensorflow as tf
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)


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

# initialis app
app = FastAPI(docs_url=None)

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css",
    )

# route
@app.get('/')
async def index():
    return{'text': 'Hello Tester'}

@app.get('/items/{name}')
async def get_item(name):
    return{'text': name}

# ml
app.get("/predict/{name}")
async def predict(name):
    image_model = model.transform([name]).toarray()
     # prep
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
    # convert to array
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    # convert to single img to batch
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index



if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)