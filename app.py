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


# --- Disease Descriptions and Solutions Dictionary ---
PLANT_DISEASE_INFO = {
    "Apple - Apple scab": {
        "description": "Fungal disease causing olive-green to brown spots on leaves, fruit, and twigs, eventually leading to defoliation and fruit distortion.",
        "solutions": [
            "**Sanitation:** Rake and destroy fallen leaves and infected fruit in the autumn to reduce overwintering spores. Prune out infected twigs during dormancy.",
            "**Fungicides:** Apply fungicides containing Myclobutanil, Captan, or Copper at bud break and continue through the growing season, especially during wet periods. Follow label instructions for timing and frequency.",
            "**Resistant Varieties:** Consider planting scab-resistant apple varieties if available in your region.",
            "**Air Circulation:** Prune trees to improve air circulation within the canopy."
        ]
    },
    "Apple - Black rot": {
        "description": "Fungal disease affecting fruit (causing black, shriveled rot), leaves (purple-bordered brown spots), and bark (cankers).",
        "solutions": [
            "**Sanitation:** Remove and destroy mummified fruit from the tree and ground. Prune out cankered branches and any dead wood.",
            "**Fungicides:** Apply fungicides like Captan or Copper at bud break and throughout the growing season, especially during warm, humid conditions.",
            "**Wound Protection:** Protect trees from mechanical injury, as wounds provide entry points for the fungus.",
            "**Pruning:** Maintain good tree structure and air circulation through proper pruning."
        ]
    },
    "Apple - Cedar apple rust": {
        "description": "Fungal disease requiring two hosts: cedar/juniper and apple. Causes bright orange-yellow spots on apple leaves and sometimes fruit.",
        "solutions": [
            "**Remove Cedars/Junipers:** The most effective long-term solution is to remove susceptible cedar or juniper trees within a few hundred feet of apple trees.",
            "**Resistant Varieties:** Plant rust-resistant apple varieties.",
            "**Fungicides:** Apply fungicides such as Myclobutanil or Chlorothalonil (on apples, check label for post-bloom use) when orange galls appear on cedars in spring, and continue until early summer.",
            "**Pruning:** Prune out galls from cedar trees before they produce spores."
        ]
    },
    "Apple - healthy": {
        "description": "The apple plant appears healthy.",
        "solutions": [
            "**Location:** Plant in full sun (at least 6-8 hours direct sunlight daily).",
            "**Soil:** Well-draining, fertile soil with a pH of 6.0-7.0.",
            "**Watering:** Consistent watering, especially during dry periods and fruit development. Deep watering is better than shallow frequent watering.",
            "**Fertilization:** Annually with a balanced fertilizer, based on soil test results.",
            "**Pruning:** Regular dormant pruning for structural integrity, air circulation, and fruit production. Summer pruning for shape and light penetration.",
            "**Pest and Disease Monitoring:** Regularly inspect trees for any signs of pests or diseases and address them promptly.",
            "**Weed Control:** Keep the area around the tree free of weeds to reduce competition for nutrients and water."
        ]
    },
    "Blueberry - healthy": {
        "description": "The blueberry plant appears healthy.",
        "solutions": [
            "**Location:** Full sun.",
            "**Soil:** Highly acidic soil (pH 4.5-5.5) rich in organic matter. Amend with peat moss, pine bark, or sulfur if needed.",
            "**Watering:** Consistent moisture, especially during fruit development. Blueberries have shallow root systems.",
            "**Mulching:** Apply a thick layer of acidic mulch (pine needles, wood chips) to conserve moisture, suppress weeds, and maintain soil acidity.",
            "**Fertilization:** Use acid-loving plant fertilizers or ammonium sulfate, especially in early spring.",
            "**Pruning:** Annual dormant pruning to remove old, unproductive canes and encourage new growth.",
            "**Pest and Disease Monitoring:** Regular inspection."
        ]
    },
    "Cherry - Powdery mildew": {
        "description": "Fungal disease causing a white, powdery growth on leaves, shoots, and sometimes fruit. Can lead to distorted growth and reduced vigor.",
        "solutions": [
            "**Air Circulation:** Prune trees to improve air circulation within the canopy.",
            "**Watering:** Avoid overhead watering, which can spread spores. Water at the base of the plant.",
            "**Fungicides:** Apply fungicides containing Myclobutanil, Sulfur (avoid on sulfur-sensitive varieties), or Potassium Bicarbonate at the first sign of symptoms and repeat as needed.",
            "**Resistant Varieties:** Choose powdery mildew-resistant cherry varieties if available.",
            "**Sanitation:** Remove and destroy heavily infected leaves and shoots."
        ]
    },
    "Cherry - healthy": {
        "description": "The cherry plant appears healthy.",
        "solutions": [
            "**Location:** Full sun.",
            "**Soil:** Well-draining, loamy soil with a pH of 6.0-7.0.",
            "**Watering:** Consistent watering, especially during dry periods and fruit development.",
            "**Fertilization:** Annually with a balanced fertilizer, based on soil test results.",
            "**Pruning:** Regular dormant pruning to maintain an open canopy, remove dead or diseased wood, and encourage fruit production.",
            "**Pest and Disease Monitoring:** Regular inspection."
        ]
    },
    "Corn (maize) - Cercospora leaf spot Gray leaf spot": {
        "description": "Fungal disease causing rectangular, gray-to-tan lesions with dark borders on leaves. Can lead to significant yield loss.",
        "solutions": [
            "**Resistant Varieties:** Plant corn hybrids with resistance to gray leaf spot.",
            "**Crop Rotation:** Rotate corn with non-host crops (e.g., soybeans, small grains) to reduce inoculum in the soil.",
            "**Tillage:** Burying crop residue can help reduce the overwintering inoculum.",
            "**Fungicides:** Apply fungicides (e.g., strobilurins, triazoles) if disease pressure is high and environmental conditions favor disease development. Consult local agricultural extension for recommended timing."
        ]
    },
    "Corn (maize) - Common rust ": {
        "description": "Fungal disease characterized by cinnamon-brown to dark brown pustules on leaves, which release powdery spores.",
        "solutions": [
            "**Resistant Varieties:** Plant corn hybrids with resistance to common rust.",
            "**Fungicides:** Fungicides can be effective if applied early in the disease development, especially on susceptible hybrids or in areas with high disease pressure."
        ]
    },
    "Corn (maize) - Northern Leaf Blight": {
        "description": "Fungal disease causing long, elliptical, grayish-green to tan lesions on leaves, often starting on lower leaves and progressing upwards.",
        "solutions": [
            "**Resistant Varieties:** Plant corn hybrids with good resistance to Northern Leaf Blight.",
            "**Crop Rotation:** Rotate corn with non-host crops.",
            "**Tillage:** Burying crop residue helps reduce inoculum.",
            "**Fungicides:** Apply fungicides when disease reaches threshold levels, especially on susceptible hybrids."
        ]
    },
    "Corn (maize) - healthy": {
        "description": "The corn (maize) plant appears healthy.",
        "solutions": [
            "**Location:** Full sun.",
            "**Soil:** Well-draining, fertile soil.",
            "**Watering:** Consistent moisture, especially during silking and kernel fill.",
            "**Fertilization:** Adequate nitrogen, phosphorus, and potassium as determined by soil test.",
            "**Spacing:** Proper spacing to allow for good air circulation.",
            "**Weed Control:** Effective weed management."
        ]
    },
    "Grape - Black rot": {
        "description": "Fungal disease causing small, circular, reddish-brown spots on leaves, and eventually shriveling, black mummified berries.",
        "solutions": [
            "**Sanitation:** Remove and destroy all mummified berries from the vine and ground. Prune out any infected tendrils or shoots.",
            "**Fungicides:** Apply fungicides (e.g., Myclobutanil, Mancozeb, Copper) according to a protective spray schedule, starting at bud break and continuing through fruit set.",
            "**Air Circulation:** Prune vines to ensure good air circulation and sunlight penetration.",
            "**Resistant Varieties:** Choose black rot-resistant grape varieties."
        ]
    },
    "Grape - Esca (Black Measles)": {
        "description": "Complex disease caused by a group of fungi. Symptoms include interveinal necrosis and tiger-stripe patterns on leaves, and internal wood discoloration (black spotting). Can lead to vine decline and death.",
        "solutions": [
            "**Pruning:** Remove diseased wood completely, cutting back to healthy tissue. Disinfect pruning tools between cuts.",
            "**Trunk Renewal:** In severe cases, cut back affected trunks below the diseased wood to encourage new shoots.",
            "**Wound Protection:** Apply wound protectants to large pruning cuts to prevent fungal entry.",
            "**Vineyard Hygiene:** Good vineyard sanitation and management to reduce stress on vines.",
            "**No Chemical Cure:** There is no direct chemical cure for established Esca. Focus on prevention and management."
        ]
    },
    "Grape - Leaf blight (Isariopsis Leaf Spot)": {
        "description": "Fungal disease causing angular, dark brown spots on leaves, often with a yellow halo. Can lead to premature defoliation.",
        "solutions": [
            "**Sanitation:** Rake and destroy fallen leaves in the autumn to reduce overwintering inoculum.",
            "**Air Circulation:** Prune vines to improve air circulation.",
            "**Fungicides:** Apply fungicides (e.g., Mancozeb, Copper-based fungicides) according to a protective spray schedule, especially during wet periods."
        ]
    },
    "Grape - healthy": {
        "description": "The grape plant appears healthy.",
        "solutions": [
            "**Location:** Full sun.",
            "**Soil:** Well-draining, moderately fertile soil.",
            "**Watering:** Consistent moisture, especially during fruit development.",
            "**Pruning:** Regular dormant pruning for vine structure, fruit production, and air circulation. Summer pruning for canopy management.",
            "**Trellising:** Provide adequate trellising or support for vines.",
            "**Fertilization:** Based on soil test results, typically a balanced fertilizer.",
            "**Pest and Disease Monitoring:** Regular inspection."
        ]
    },
    "Orange - Haunglongbing (Citrus greening)": {
        "description": "Bacterial disease spread by the Asian citrus psyllid. Causes yellowing of leaves in blotchy patterns, stunted growth, lopsided fruit, and bitter taste. Fatal to trees.",
        "solutions": [
            "**Vector Control:** Control the Asian citrus psyllid through insecticide applications (systemic and foliar) in affected areas.",
            "**Remove Infected Trees:** Immediately remove and destroy infected trees to prevent further spread.",
            "**Quarantine:** Adhere to local and national quarantines to prevent the movement of infected plant material.",
            "**Resistant Varieties:** Research and plant any newly developed tolerant or resistant varieties.",
            "**No Cure:** There is currently no cure for HLB. Prevention and vector control are paramount."
        ]
    },
    "Peach - Bacterial spot": {
        "description": "Bacterial disease causing small, angular, water-soaked spots on leaves that turn dark brown/black, often with a purplish halo. Causes spots on fruit and cankers on twigs.",
        "solutions": [
            "**Resistant Varieties:** Plant peach varieties with genetic resistance to bacterial spot.",
            "**Copper Sprays:** Apply dormant copper sprays in late fall or early spring. Repeated applications during the growing season may be needed, but can cause phytotoxicity.",
            "**Pruning:** Prune out severely infected twigs during dormancy.",
            "**Avoid Overhead Watering:** Water at the base of the tree to reduce leaf wetness.",
            "**Minimize Wounds:** Avoid practices that create wounds on the tree."
        ]
    },
    "Peach - healthy": {
        "description": "The peach plant appears healthy.",
        "solutions": [
            "**Location:** Full sun.",
            "**Soil:** Well-draining, loamy soil with a pH of 6.0-7.0.",
            "**Watering:** Consistent watering, especially during fruit development.",
            "**Fertilization:** Annually with a balanced fertilizer, based on soil test results.",
            "**Pruning:** Regular dormant pruning for an open vase shape to promote air circulation and fruit production. Thinning fruit is also crucial.",
            "**Pest and Disease Monitoring:** Regular inspection and timely intervention."
        ]
    },
    "Pepper, bell - Bacterial spot": {
        "description": "Bacterial disease causing small, circular, water-soaked spots on leaves, stems, and fruit. Spots on leaves turn brown with yellow halos; on fruit, they are raised and scab-like.",
        "solutions": [
            "**Resistant Varieties:** Plant bell pepper varieties with resistance to bacterial spot.",
            "**Sanitation:** Remove and destroy infected plant debris. Avoid working with plants when wet.",
            "**Seed Treatment:** Use disease-free seeds or treat seeds with hot water.",
            "**Copper Sprays:** Apply copper-based fungicides/bactericides as a preventative measure, but resistance can develop.",
            "**Crop Rotation:** Rotate peppers with non-host crops.",
            "**Spacing:** Provide adequate spacing for air circulation."
        ]
    },
    "Pepper, bell - healthy": {
        "description": "The bell pepper plant appears healthy.",
        "solutions": [
            "**Location:** Full sun.",
            "**Soil:** Well-draining, fertile soil rich in organic matter.",
            "**Watering:** Consistent and even moisture. Avoid waterlogging.",
            "**Fertilization:** Balanced fertilizer, especially during flowering and fruiting.",
            "**Support:** Provide stakes or cages for support as fruit develops.",
            "**Pest and Disease Monitoring:** Regular inspection."
        ]
    },
    "Potato - Early blight": {
        "description": "Fungal disease causing dark brown to black spots on leaves, often with concentric rings (target-like appearance). Primarily affects older leaves.",
        "solutions": [
            "**Resistant Varieties:** Plant early blight-resistant potato varieties.",
            "**Crop Rotation:** Rotate potatoes with non-solanaceous crops for at least 2-3 years.",
            "**Sanitation:** Remove and destroy infected plant debris at the end of the season.",
            "**Fungicides:** Apply fungicides containing Chlorothalonil or Mancozeb as a preventative measure when disease pressure is high.",
            "**Watering:** Avoid overhead watering. Water at the base of the plants.",
            "**Plant Spacing:** Ensure adequate spacing for air circulation."
        ]
    },
    "Potato - Late blight": {
        "description": "Oomycete disease (not a true fungus) causing irregular, water-soaked lesions on leaves that rapidly turn brown/black, often with a fuzzy white growth on the undersides in humid conditions. Can quickly decimate crops. Affects tubers as well.",
        "solutions": [
            "**Resistant Varieties:** Plant late blight-resistant potato varieties.",
            "**Sanitation:** Do not plant infected tubers. Destroy all volunteers and cull piles.",
            "**Fungicides:** Apply highly effective systemic and protectant fungicides (e.g., those containing Propamocarb, Mancozeb, Chlorothalonil, or Fluazinam) on a regular schedule, especially during cool, wet weather.",
            "**Hilling:** Hill soil around plants to prevent spores from reaching tubers.",
            "**Air Circulation:** Ensure good air circulation.",
            "**Remove Infected Plants:** Promptly remove and destroy any infected plants."
        ]
    },
    "Potato - healthy": {
        "description": "The potato plant appears healthy.",
        "solutions": [
            "**Location:** Full sun.",
            "**Soil:** Loose, well-drraining, slightly acidic to neutral soil.",
            "**Watering:** Consistent and even moisture, especially during tuber development.",
            "**Fertilization:** Balanced fertilizer, with emphasis on potassium.",
            "**Hilling:** Regularly hill soil around plants to protect developing tubers from light and disease.",
            "**Crop Rotation:** Practice good crop rotation.",
            "**Disease-Free Seed:** Use certified disease-free seed potatoes."
        ]
    },
    "Raspberry - healthy": {
        "description": "The raspberry plant appears healthy.",
        "solutions": [
            "**Location:** Full sun to partial shade.",
            "**Soil:** Well-draining, loamy soil with a pH of 6.0-6.5.",
            "**Watering:** Consistent moisture, especially during fruiting.",
            "**Support:** Provide trellising or support for canes.",
            "**Pruning:** Regular pruning to remove old fruiting canes (floricanes) and thin new canes (primocanes) for better air circulation and light penetration.",
            "**Mulching:** Apply mulch to conserve moisture and suppress weeds.",
            "**Fertilization:** Annually with a balanced fertilizer, based on soil test results."
        ]
    },
    "Soybean - healthy": {
        "description": "The soybean plant appears healthy.",
        "solutions": [
            "**Location:** Full sun.",
            "**Soil:** Well-draining, fertile soil with appropriate pH.",
            "**Watering:** Adequate moisture, especially during critical growth stages (flowering and pod fill).",
            "**Crop Rotation:** Essential for managing soil-borne diseases and pests.",
            "**Seed Treatment:** Use high-quality, treated seeds.",
            "**Nutrient Management:** Proper fertilization based on soil tests.",
            "**Weed Control:** Effective weed management."
        ]
    },
    "Squash - Powdery mildew": {
        "description": "Fungal disease causing a white, powdery growth on the surface of leaves and stems. Can lead to premature defoliation and reduced fruit quality.",
        "solutions": [
            "**Resistant Varieties:** Plant squash varieties with resistance to powdery mildew.",
            "**Air Circulation:** Space plants adequately to promote good air circulation.",
            "**Watering:** Water at the base of the plants to avoid wetting leaves.",
            "**Fungicides:** Apply fungicides like Neem oil, Sulfur (avoid on sulfur-sensitive varieties), or Potassium Bicarbonate at the first sign of symptoms. Biological fungicides containing *Bacillus subtilis* can also be effective.",
            "**Sanitation:** Remove and destroy infected leaves or entire plants at the end of the season."
        ]
    },
    "Strawberry - Leaf scorch": {
        "description": "Fungal disease causing purplish to reddish-brown spots on leaves that enlarge and coalesce, causing the leaf edges to 'scorch' and curl upwards. Can affect fruit development.",
        "solutions": [
            "**Sanitation:** Remove and destroy infected leaves and plant debris.",
            "**Air Circulation:** Ensure good air circulation by proper plant spacing and removing excess runners.",
            "**Fungicides:** Apply fungicides containing Myclobutanil or Chlorothalonil as a preventative measure, especially during wet periods.",
            "**Resistant Varieties:** Plant leaf scorch-resistant strawberry varieties.",
            "**Watering:** Water at the base of plants."
        ]
    },
    "Strawberry - healthy": {
        "description": "The strawberry plant appears healthy.",
        "solutions": [
            "**Location:** Full sun.",
            "**Soil:** Well-draining, slightly acidic soil (pH 5.5-6.5) rich in organic matter.",
            "**Watering:** Consistent and even moisture, especially during flowering and fruiting.",
            "**Mulching:** Apply straw mulch around plants to conserve moisture, suppress weeds, and keep fruit clean.",
            "**Runner Management:** Control runners to prevent overcrowding and maintain plant vigor.",
            "**Fertilization:** Balanced fertilizer, especially after harvest.",
            "**Renovation:** Consider renovating June-bearing plants after harvest for better production next year."
        ]
    },
    "Tomato - Bacterial spot": {
        "description": "Bacterial disease causing small, circular, water-soaked spots on leaves, stems, and fruit. Spots on leaves turn dark brown/black with yellow halos; on fruit, they are raised, scab-like, and often have a white halo.",
        "solutions": [
            "**Resistant Varieties:** Plant bacterial spot-resistant tomato varieties.",
            "**Sanitation:** Remove and destroy infected plant debris. Avoid working with plants when wet.",
            "**Seed Treatment:** Use disease-free seeds or treat seeds with hot water.",
            "**Copper Sprays:** Apply copper-based bactericides preventatively. Can cause phytotoxicity.",
            "**Crop Rotation:** Rotate tomatoes with non-solanaceous crops.",
            "**Spacing:** Provide adequate spacing for air circulation."
        ]
    },
    "Tomato - Early blight": {
        "description": "Fungal disease causing dark brown to black spots on leaves, often with concentric rings (target-like appearance). Primarily affects older leaves.",
        "solutions": [
            "**Resistant Varieties:** Plant early blight-resistant tomato varieties.",
            "**Crop Rotation:** Rotate tomatoes with non-solanaceous crops for at least 2-3 years.",
            "**Sanitation:** Remove and destroy infected plant debris at the end of the season. Prune off lower, older leaves that touch the soil.",
            "**Fungicides:** Apply fungicides containing Chlorothalonil or Mancozeb as a preventative measure when disease pressure is high.",
            "**Watering:** Avoid overhead watering. Water at the base of the plants.",
            "**Mulching:** Apply mulch to prevent splashing of spores from soil to leaves."
        ]
    },
    "Tomato - Late blight": {
        "description": "Oomycete disease (not a true fungus) causing irregular, water-soaked lesions on leaves that rapidly turn brown/black, often with a fuzzy white growth on the undersides in humid conditions. Can quickly decimate crops. Affects stems and fruit.",
        "solutions": [
            "**Resistant Varieties:** Plant late blight-resistant tomato varieties.",
            "**Sanitation:** Do not plant infected transplants. Destroy all volunteer potato and tomato plants.",
            "**Fungicides:** Apply highly effective systemic and protectant fungicides (e.g., those containing Propamocarb, Mancozeb, Chlorothalonil, or Fluazinam) on a regular schedule, especially during cool, wet weather.",
            "**Air Circulation:** Ensure good air circulation.",
            "**Remove Infected Plants:** Promptly remove and destroy any infected plants."
        ]
    },
    "Tomato - Leaf Mold": {
        "description": "Fungal disease causing pale green or yellow spots on the upper leaf surface, with olive-green to brown velvety patches on the corresponding undersides.",
        "solutions": [
            "**Air Circulation:** Good ventilation in greenhouses and proper spacing in gardens. Prune to improve airflow.",
            "**Humidity Control:** Reduce humidity levels if growing in a greenhouse.",
            "**Resistant Varieties:** Plant leaf mold-resistant tomato varieties.",
            "**Fungicides:** Some fungicides (e.g., copper-based) can offer limited control, but cultural practices are key.",
            "**Sanitation:** Remove infected leaves."
        ]
    },
    "Tomato - Septoria leaf spot": {
        "description": "Fungal disease causing numerous small, circular spots with dark brown borders and tan/gray centers, often with tiny black dots (fruiting bodies) in the center of the spot. Primarily affects lower leaves.",
        "solutions": [
            "**Sanitation:** Remove and destroy infected lower leaves. Clean up all plant debris at the end of the season.",
            "**Watering:** Water at the base of the plants to avoid splashing spores.",
            "**Mulching:** Apply mulch to prevent splashing from soil to leaves.",
            "**Fungicides:** Apply fungicides containing Chlorothalonil or Mancozeb as a preventative measure.",
            "**Staking/Caging:** Keep plants off the ground.",
            "**Crop Rotation:** Rotate tomatoes."
        ]
    },
    "Tomato - Spider mites Two-spotted spider mite": {
        "description": "Tiny arachnids that feed on plant sap, causing stippling (tiny yellow/white dots) on leaves, bronze discoloration, and fine webbing, especially on the undersides of leaves.",
        "solutions": [
            "**Water Spray:** Strong spray of water to dislodge mites, especially on the undersides of leaves.",
            "**Horticultural Oil/Insecticidal Soap:** Apply horticultural oils or insecticidal soaps, ensuring thorough coverage of both sides of leaves.",
            "**Predatory Mites:** Introduce natural predators like predatory mites (*Phytoseiulus persimilis*) in enclosed environments.",
            "**Humidity:** Increase humidity if possible, as spider mites prefer dry conditions.",
            "**Remove Infested Leaves:** Remove and destroy heavily infested leaves."
        ]
    },
    "Tomato - Target Spot": {
        "description": "Fungal disease causing dark brown spots on leaves, often with concentric rings, similar to early blight but typically darker and more irregular. Can also affect stems and fruit.",
        "solutions": [
            "**Sanitation:** Remove and destroy infected plant debris.",
            "**Crop Rotation:** Rotate tomatoes with non-solanaceous crops.",
            "**Fungicides:** Apply fungicides containing Chlorothalonil or Mancozeb."
        ]
    },
    "Tomato - Tomato Yellow Leaf Curl Virus": {
        "description": "Viral disease transmitted by whiteflies. Causes severe stunting, upward curling and yellowing of leaf margins, and reduced fruit set.",
        "solutions": [
            "**Whitefly Control:** Control whitefly populations using insecticides (e.g., neonicotinoids, pyrethroids) or biological controls (e.g., predatory insects, parasitic wasps).",
            "**Resistant Varieties:** Plant tomato varieties with genetic resistance or tolerance to TYLCV.",
            "**Remove Infected Plants:** Immediately remove and destroy infected plants to prevent further spread.",
            "**Row Covers:** Use fine mesh row covers to exclude whiteflies.",
            "**Weed Control:** Control weeds that can host whiteflies or the virus."
        ]
    },
    "Tomato - Tomato mosaic virus": {
        "description": "Viral disease causing mosaic patterns (light and dark green patches) on leaves, leaf distortion, stunting, and reduced fruit production.",
        "solutions": [
            "**Sanitation:** Wash hands and disinfect tools thoroughly when working with tomatoes, as the virus is highly transmissible mechanically.",
            "**Resistant Varieties:** Plant tomato varieties resistant to TMV.",
            "**Remove Infected Plants:** Promptly remove and destroy infected plants.",
            "**Avoid Tobacco Products:** Avoid smoking or using tobacco products around tomatoes, as tobacco can carry the virus.",
            "**Weed Control:** Control weeds that may host the virus."
        ]
    },
    "Tomato - healthy": {
        "description": "The tomato plant appears healthy.",
        "solutions": [
            "**Location:** Full sun (at least 6-8 hours direct sunlight daily).",
            "**Soil:** Well-draining, fertile soil rich in organic matter, with a pH of 6.0-6.8.",
            "**Watering:** Consistent, deep watering at the base of the plant, especially during flowering and fruiting. Avoid overhead watering.",
            "**Staking/Caging:** Provide strong support (stakes, cages, or trellises) for plants to keep fruit and foliage off the ground, improving air circulation.",
            "**Pruning:** Remove suckers (side shoots) for indeterminate varieties to direct energy to fruit production. Remove lower leaves touching the soil.",
            "**Mulching:** Apply a layer of mulch (straw, shredded leaves) to conserve moisture, suppress weeds, and prevent soil splash.",
            "**Fertilization:** Balanced fertilizer, with emphasis on phosphorus and potassium during flowering and fruiting.",
            "**Crop Rotation:** Rotate tomatoes with non-solanaceous crops to reduce soil-borne diseases."
        ]
    },
}


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
    uvicorn.run(app, host="0.0.0.0", port=8000) 