import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os

# Estilo de la página
st.set_page_config(page_title="Verificación de Seguridad SST", page_icon="🦺")

# Cargar clases desde el archivo
def cargar_clases():
    try:
        with open("clasesSST.txt", "r", encoding="utf-8") as f:
            return [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        st.error("Archivo clasesSST.txt no encontrado.")
        return []

CLASES = cargar_clases()

# Cargar modelo TFLite
@st.cache_resource
def cargar_modelo():
    interpreter = tf.lite.Interpreter(model_path="yolov8n_float32.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = cargar_modelo()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Preprocesar imagen
def preprocesar(imagen):
    imagen = imagen.resize((640, 640))
    img_array = np.array(imagen).astype(np.float32)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Dibujar resultados
def dibujar_cajas(imagen, cajas, clases, puntuaciones, umbral=0.3):
    imagen = np.array(imagen)
    h, w, _ = imagen.shape
    for i in range(len(puntuaciones)):
        if puntuaciones[i] > umbral:
            y1, x1, y2, x2 = cajas[i]
            x1, x2 = int(x1 * w), int(x2 * w)
            y1, y2 = int(y1 * h), int(y2 * h)
            class_id = int(clases[i])
            label = CLASES[class_id] if class_id < len(CLASES) else f"ID {class_id}"
            cv2.rectangle(imagen, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(imagen, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return imagen

# Título y descripción
st.title("🦺 Verificación de Implementos de Seguridad")
st.write("Esta aplicación detecta si una persona porta elementos de seguridad como casco, chaleco, gafas, etc.")

# Carga de imagen

img_input = st.camera_input("Captura una imagen") or st.file_uploader("O carga una imagen", type=["jpg", "png", "jpeg"]) or (lambda url: BytesIO(requests.get(url).content) if url else None)(st.text_input("O ingresa una URL de imagen"))

if img_input:
    imagen = Image.open(img_input)
    st.image(imagen, caption="Imagen cargada", use_container_width=True)

    input_data = preprocesar(imagen)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Suponiendo salida: [boxes, clases, scores]
    try:
        cajas = interpreter.get_tensor(output_details[0]['index'])[0]
        clases = interpreter.get_tensor(output_details[1]['index'])[0]
        puntuaciones = interpreter.get_tensor(output_details[2]['index'])[0]
    except:
        st.error("No se pudieron interpretar las salidas del modelo.")
        st.stop()

    imagen_salida = dibujar_cajas(imagen, cajas, clases, puntuaciones, umbral=0.3)
    st.image(imagen_salida, caption="Resultados de detección", use_container_width=True)

    # Mostrar objetos detectados
    detectados = [CLASES[int(clases[i])] for i in range(len(puntuaciones)) if puntuaciones[i] > 0.3]
    if detectados:
        st.success("Implementos detectados:")
        st.write(", ".join(set(detectados)))
    else:
        st.warning("No se detectaron implementos de seguridad.")
