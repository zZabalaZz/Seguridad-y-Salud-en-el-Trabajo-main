import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

# Cargar clases
with open("clasesSST.txt", "r") as f:
    CLASSES = [line.strip() for line in f.readlines()]

# Cargar modelo TFLite
interpreter = tf.lite.Interpreter(model_path="yolov8n_float32.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape'][1:3]

# Preprocesamiento de imagen
def preprocess_image(image):
    image_resized = image.resize(input_shape)
    image_array = np.asarray(image_resized).astype(np.float32)
    image_array = np.expand_dims(image_array, axis=0)  # [1, h, w, 3]
    return image_array

# Postprocesamiento para YOLOv8 salida [1, N, 6]
def postprocess_output(output_data, original_image):
    w_img, h_img = original_image.size
    boxes, class_ids, scores = [], [], []

    detections = output_data[0][0]  # [1, N, 6]

    for det in detections:
        if det[4] > 0.4:  # Umbral de confianza
            x_center, y_center, width, height = det[0], det[1], det[2], det[3]
            class_id = int(det[5])
            
            # Coordenadas absolutas
            left = int((x_center - width / 2) * w_img)
            top = int((y_center - height / 2) * h_img)
            right = int((x_center + width / 2) * w_img)
            bottom = int((y_center + height / 2) * h_img)

            boxes.append((left, top, right, bottom))
            class_ids.append(class_id)
            scores.append(det[4])

    return boxes, class_ids, scores

# Interfaz Streamlit
st.title("🦺 Detección de Elementos de Seguridad con YOLOv8")

uploaded_file = st.file_uploader("📸 Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Imagen Original", use_column_width=True)

    # Inferencia
    input_tensor = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output_data = [interpreter.get_tensor(output['index']) for output in output_details]

    # Postprocesamiento
    boxes, class_ids, scores = postprocess_output(output_data, image.copy())
    draw = ImageDraw.Draw(image)
    detected_labels = set()

    for box, cls_id, score in zip(boxes, class_ids, scores):
        label = CLASSES[cls_id] if cls_id < len(CLASSES) else f"Clase {cls_id}"
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0], box[1]), f"{label} ({score:.2f})", fill="red")
        detected_labels.add(label)

    st.image(image, caption="🟥 Imagen con Detecciones", use_column_width=True)

    st.markdown("### ✅ Objetos detectados:")
    if detected_labels:
        for label in detected_labels:
            st.write(f"- {label}")
    else:
        st.write("⚠️ No se detectaron elementos con suficiente confianza.")