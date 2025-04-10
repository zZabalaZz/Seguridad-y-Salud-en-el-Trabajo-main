import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
import io

# Cargar clases
with open("clasesSST.txt", "r") as f:
    CLASSES = [line.strip() for line in f.readlines()]

# Cargar modelo TFLite
interpreter = tf.lite.Interpreter(model_path="yolov8n_float32.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape'][1:3]

# Función para preparar imagen
def preprocess_image(image):
    image_resized = image.resize(input_shape)
    image_array = np.asarray(image_resized).astype(np.float32)
    image_array = np.expand_dims(image_array, axis=0)  # [1, h, w, 3]
    return image_array

# Función para post-procesar salidas
def postprocess_output(output_data, original_image):
    boxes, class_ids, scores = [], [], []
    
    output = output_data[0][0]  # Dependiendo del modelo puede necesitar ajuste

    h, w = original_image.size
    for det in output_data[0]:
        if det[4] > 0.4:  # Umbral de confianza
            x, y, width, height = det[0], det[1], det[2], det[3]
            left = int((x - width / 2) * w)
            top = int((y - height / 2) * h)
            right = int((x + width / 2) * w)
            bottom = int((y + height / 2) * h)
            class_id = int(det[5])
            boxes.append((left, top, right, bottom))
            class_ids.append(class_id)
            scores.append(det[4])
    
    return boxes, class_ids, scores

# Interfaz Streamlit
st.title("Detección de Elementos de Seguridad con YOLOv8")
uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen original", use_column_width=True)

    input_tensor = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()

    output_data = [interpreter.get_tensor(output['index']) for output in output_details]
    boxes, class_ids, scores = postprocess_output(output_data, image)

    draw = ImageDraw.Draw(image)
    detected_labels = set()

    for box, cls_id, score in zip(boxes, class_ids, scores):
        label = CLASSES[cls_id] if cls_id < len(CLASSES) else f"Clase {cls_id}"
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0], box[1]), f"{label} ({score:.2f})", fill="red")
        detected_labels.add(label)

    st.image(image, caption="Imagen con Detecciones", use_column_width=True)

    st.markdown("### Objetos detectados:")
    if detected_labels:
        for label in detected_labels:
            st.write(f"- {label}")
    else:
        st.write("No se detectaron elementos con suficiente confianza.")
