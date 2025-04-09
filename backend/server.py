from flask import Flask, request, send_file, jsonify
from io import BytesIO
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocessor, decode_predictions as resnet_decode
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocessor, decode_predictions as vgg_decode
from PIL import Image
import numpy as np

app = Flask(__name__)
CORS(app)

resnet_model = ResNet50(weights='imagenet')
vgg_model = VGG16(weights='imagenet')
# unet_model = load_model("./Stellens_Lip_Detector_final.keras") # TODO: CHANGE THIS!

@app.route("/api/predict", methods=["POST"])
def predict():
    if 'image' not in request.files or 'model' not in request.form:
        return jsonify({"error": "Missing image or model parameter"}), 400

    model_name = request.form["model"]
    image_file = request.files["image"]
    image = Image.open(image_file).convert("RGB")

    image = image.resize((224, 224))
    image_array = img_to_array(image)  
    image_array = np.expand_dims(image_array, axis=0) 

    if model_name == "resnet":
        img = resnet_preprocessor(image_array)
        preds = resnet_model.predict(img)
        result = resnet_decode(preds, top=1)[0][0][1] # [[('n02099712', 'Labrador_retriever', 0.74605143)]]
        print(resnet_decode(preds, top=1)) 
        print(preds)
        return jsonify({"message": result}), 200

    elif model_name == "vgg":
        img = vgg_preprocessor(image_array)
        preds = vgg_model.predict(img)
        result = vgg_decode(preds, top=1)[0][0][1]
        return jsonify({"message": result}), 200

    # elif model_name == "unet":
    #     image = image.resize((28, 28, 1)) 
    #     img_np = np.array(image) / 255.0
    #     img_np = np.expand_dims(img_np, axis=0)
    #     mask = unet_model.predict(img_np)[0]
    #     mask = (mask > 0.5).astype(np.uint8) * 255 

    #     mask_img = Image.fromarray(mask.squeeze())
    #     buf = BytesIO()
    #     mask_img.save(buf, format='PNG')
    #     buf.seek(0)
    #     return send_file(buf, mimetype='image/png'), 200

    print("error occured here. No model found!")
    return jsonify({"error": "Unsupported model name"}), 400


if __name__ == "__main__" :
    app.run(debug=False, port=5000)