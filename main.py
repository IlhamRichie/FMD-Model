from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Definisi ulang CBAM dengan perbaikan
from tensorflow.keras.layers import Layer, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Multiply, Concatenate, Activation
import tensorflow.keras.backend as K

class CBAM(Layer):
    def __init__(self, filters, reduction_ratio=8, **kwargs):
        super(CBAM, self).__init__(**kwargs)
        self.filters = filters
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        self.shared_dense_one = Dense(self.filters // self.reduction_ratio, activation='relu')
        self.shared_dense_two = Dense(self.filters)
        self.conv_spatial = Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')

    def call(self, inputs):
        avg_pool = GlobalAveragePooling2D()(inputs)
        max_pool = GlobalMaxPooling2D()(inputs)

        avg_pool = self.shared_dense_two(self.shared_dense_one(K.reshape(avg_pool, (-1, 1, 1, self.filters))))
        max_pool = self.shared_dense_two(self.shared_dense_one(K.reshape(max_pool, (-1, 1, 1, self.filters))))

        channel_attention = Activation('sigmoid')(avg_pool + max_pool)
        channel_refined = Multiply()([inputs, channel_attention])

        avg_pool_spatial = tf.reduce_mean(channel_refined, axis=-1, keepdims=True)
        max_pool_spatial = tf.reduce_max(channel_refined, axis=-1, keepdims=True)
        spatial_attention = Concatenate(axis=-1)([avg_pool_spatial, max_pool_spatial])
        spatial_attention = self.conv_spatial(spatial_attention)

        refined_feature = Multiply()([channel_refined, spatial_attention])
        return refined_feature

# Flask app
app = Flask(__name__)

# Load model dengan custom_objects dan compile=False
model = tf.keras.models.load_model("newCBAM-DenseNet.h5", custom_objects={"CBAM": CBAM}, compile=False)

# Mapping indeks kelas ke label PMK/FMD
class_mapping = {
    0: "PMK Parah",
    1: "PMK Sedang",
    2: "PMK Ringan",
    3: "Sehat",
}

# Preprocess image
def preprocess_image(image):
    image = image.resize((150, 150))  # Sesuai ukuran input model
    image = np.array(image) / 255.0  # Normalisasi ke [0, 1]
    image = np.expand_dims(image, axis=0)  # Tambahkan dimensi batch
    return image

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        # Buka gambar
        image = Image.open(io.BytesIO(file.read()))

        # Preprocess gambar
        processed_image = preprocess_image(image)

        # Prediksi dengan model
        predictions = model.predict(processed_image)

        # Ambil kelas dan probabilitas tertinggi
        predicted_class_index = int(np.argmax(predictions, axis=1)[0])  # Indeks kelas prediksi
        predicted_probability = float(np.max(predictions, axis=1)[0])  # Probabilitas prediksi

        # Dapatkan label kategorikal dari mapping
        predicted_class_label = class_mapping.get(predicted_class_index, "Unknown")

        # Format hasil
        result = {
            "predicted_class_index": predicted_class_index,
            "predicted_class_label": predicted_class_label,
            "predicted_probability": predicted_probability,
            "predictions": predictions.tolist()  # Output lengkap dari model
        }

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, threaded=True)