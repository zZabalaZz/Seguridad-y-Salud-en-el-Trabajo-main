import streamlit as st
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw

# Cargar clases
with open("clasesSST.txt", "r") as f:
    CLASSES = [line.strip() for line in f.readlines()]

# Configurar sesión de ONNX
onnx_model_path = "yolov8n.onnx"
session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape  # [1, 3, h, w]
input_height, input_width = input_shape[2], input_shape[3]

# Preprocesar imagen para YOLOv8 ONNX
def preprocess_image(image):
    image_resized = image.resize((input_width, input_height))
    image_array = np.array(image_resized).astype(np.float32) / 255.0  # Normalizar
    image_array = np.transpose(image_array, (2, 0, 1))  # HWC -> CHW
    image_array = np.expand_dims(image_array, axis=0)  # [1, 3, h, w]
    return image_array

# Postprocesamiento para YOLOv8 ONNX (salida [1, N, 6])
def postprocess_output(output, orig_image, conf_threshold=0.3):
    image_width, image_height = orig_image.size
    detections = output[0]  # [1, N, 6]
    boxes, class_ids, scores = [], [], []

    for det in detections:
        if det[4] > conf_threshold:
            x_center, y_center, width, height = det[0], det[1], det[2], det[3]
            class_id = int(det[5])

            # Convertir a coordenadas absolutas
            left = int((x_center - width / 2) * image_width)
            top = int((y_center - height / 2) * image_height)
            right = int((x_center + width / 2) * image_width)
            bottom = int((y_center + height / 2) * image_height)

            boxes.append((left, top, right, bottom))
            class_ids.append(class_id)
            scores.append(det[4])

    return boxes, class_ids, scores

# Interfaz Streamlit
st.title("🦺 Detección de Seguridad con YOLOv8 ONNX")

uploaded_file = st.file_uploader("📷 Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Imagen Original", use_column_width=True)

    # Inferencia
    input_tensor = preprocess_image(image)
    output = session.run(None, {input_name: input_tensor})
    boxes, class_ids, scores = postprocess_output(output, image.copy())

    # Dibujar resultados
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
