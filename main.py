from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
from tensorflow.keras.layers import Layer, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Multiply, Concatenate, Activation
import tensorflow.keras.backend as K
import cv2

# Definisi ulang CBAM
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

# Load model dengan custom_objects
with tf.keras.utils.custom_object_scope({'CBAM': CBAM}):
    model = tf.keras.models.load_model("newCBAM-DenseNet.h5", compile=False)

# Mapping indeks kelas ke label
class_mapping = {
    0: "PMK",
    1: "Sehat",
}

# Preprocess image
def preprocess_image(image):
    image = image.resize((150, 150))  # Sesuaikan dengan input model
    image = np.array(image) / 255.0  # Normalisasi
    image = np.expand_dims(image, axis=0)  # Tambahkan dimensi batch
    return image

# Generate Grad-CAM heatmap
def generate_grad_cam(model, image, layer_name="conv5_block32_concat"):  # Ganti dengan nama layer convolutional terakhir
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        loss = predictions[:, np.argmax(predictions[0])]

    grads = tape.gradient(loss, conv_outputs)[0]
    output = conv_outputs[0]

    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = np.zeros(output.shape[0:2], dtype=np.float32)

    for index, w in enumerate(weights):
        cam += w * output[:, :, index]

    cam = cv2.resize(cam.numpy(), (150, 150))
    cam = np.maximum(cam, 0)
    heatmap = (cam - cam.min()) / (cam.max() - cam.min())
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap

# Overlay heatmap pada gambar asli
def overlay_heatmap(image, heatmap, alpha=0.5):
    # Ubah gambar asli ke format BGR (OpenCV)
    image = np.array(image)
    if image.shape[-1] == 4:  # Jika gambar memiliki channel alpha (RGBA)
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Resize heatmap ke ukuran gambar asli
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Gabungkan heatmap dengan gambar asli
    overlayed = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    return overlayed

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
        predicted_class_index = int(np.argmax(predictions, axis=1)[0])
        predicted_probability = float(np.max(predictions, axis=1)[0])

        # Dapatkan label kategorikal
        predicted_class_label = class_mapping.get(predicted_class_index, "Unknown")

        # Generate Grad-CAM heatmap
        heatmap = generate_grad_cam(model, processed_image, "conv5_block32_concat")  # Ganti dengan nama layer convolutional terakhir

        # Overlay heatmap pada gambar asli
        overlayed_image = overlay_heatmap(image, heatmap, alpha=0.5)  # Atur transparansi di sini
        overlayed_image = Image.fromarray(cv2.cvtColor(overlayed_image, cv2.COLOR_BGR2RGB))

        # Simpan gambar original dan heatmap
        original_image_bytes = io.BytesIO()
        image.save(original_image_bytes, format="JPEG")
        original_image_base64 = base64.b64encode(original_image_bytes.getvalue()).decode("utf-8")

        overlayed_bytes = io.BytesIO()
        overlayed_image.save(overlayed_bytes, format="JPEG")
        overlayed_base64 = base64.b64encode(overlayed_bytes.getvalue()).decode("utf-8")

        # Format hasil
        result = {
            "predicted_class_index": predicted_class_index,
            "predicted_class_label": predicted_class_label,
            "predicted_probability": predicted_probability,
            "original_image": original_image_base64,
            "overlayed_image": overlayed_base64,  # Gambar dengan overlay heatmap
        }

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
