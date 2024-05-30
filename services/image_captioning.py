from flask import Flask, request, jsonify
import tensorflow_hub as hub
import numpy as np
import tensorflow_text  # Required for TF-Hub models using text inputs
import PIL.Image as Image

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained image captioning model from TensorFlow Hub
hub_handle = 'https://tfhub.dev/google/lucid/imagenet/inceptionv1-2016-03-01/feature-vector/1'
image_model = hub.load(hub_handle)

# Function to preprocess image
def preprocess_image(image_path):
    img = Image.open(image_path).resize((224, 224))
    img = np.array(img) / 255.0  # Normalize pixel values
    return img[np.newaxis, ...]

# Function to generate caption for image
def generate_caption(image_path):
    img = preprocess_image(image_path)
    features = image_model(img)
    return features.numpy().tolist()  # Convert to list for JSON serialization

# Define API endpoint for image captioning
@app.route('/generate_caption', methods=['POST'])
def generate_caption_api():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found'})
    img_file = request.files['image']
    img_path = 'temp_img.jpg'
    img_file.save(img_path)
    features = generate_caption(img_path)
    return jsonify({'features': features})

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
