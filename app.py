import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import google.generativeai as genai

# --- 1. CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Asistente Radiológico IA", page_icon="🫁", layout="wide")

st.title("🫁 Asistente de Diagnóstico por IA: Radiografía de Tórax")
st.markdown("Sube una radiografía frontal de tórax para obtener un pre-diagnóstico con análisis de incertidumbre y explicabilidad visual.")

# --- 2. DEFINICIÓN DEL MODELO (Igual que en Kaggle) ---
class DenseNet121_MCDropout(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(DenseNet121_MCDropout, self).__init__()
        self.densenet = models.densenet121(weights=None) # Ya no bajamos de internet, cargaremos nuestros pesos
        in_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        return self.densenet(x)
        
    def enable_dropout(self):
        for m in self.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

# --- 3. FUNCIONES CACHEADAS (Para que la app sea rápida) ---
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DenseNet121_MCDropout()
    # Cargar los pesos que entrenaste en Kaggle
    model.load_state_dict(torch.load('densenet_pneumonia_mc.pth', map_location=device))
    model.to(device)
    return model, device

model, device = load_model()

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)

def mc_dropout_predict(model, image_tensor, num_passes=20):
    model.eval()
    model.enable_dropout()
    predictions = []
    with torch.no_grad():
        for _ in range(num_passes):
            output = model(image_tensor)
            prob = torch.sigmoid(output).item()
            predictions.append(prob)
    return np.mean(predictions), np.std(predictions)

def generate_gradcam(model, image_tensor, original_image):
    target_layers = [model.densenet.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=image_tensor, targets=None)[0, :]
    
    # Redimensionar la imagen original al tamaño de la red (224x224)
    img_resized = original_image.resize((224, 224))
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    
    # Si la imagen es en escala de grises, convertir a RGB
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        
    visualization = show_cam_on_image(img_array, grayscale_cam, use_rgb=True)
    return visualization

# --- 4. BARRA LATERAL (Configuración y API Key) ---
with st.sidebar:
    st.header("⚙️ Configuración")
    gemini_key = st.text_input("Ingresa tu Gemini API Key:", type="password", help="Obtenla gratis en Google AI Studio")
    st.markdown("---")
    st.info("**Nota Técnica:** Este modelo cuantifica su propia incertidumbre mediante aproximación Bayesiana (Monte Carlo Dropout).")

# --- 5. INTERFAZ PRINCIPAL ---
uploaded_file = st.file_uploader("Selecciona una radiografía (JPEG/PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mostrar imágenes en dos columnas
    col1, col2 = st.columns(2)
    
    original_image = Image.open(uploaded_file).convert('RGB')
    
    with col1:
        st.subheader("Radiografía Original")
        st.image(original_image, use_container_width=True)
        
    if st.button("Generar Diagnóstico y Reporte", type="primary"):
        with st.spinner('Analizando imagen neuronalmente...'):
            # 1. Procesar
            img_tensor = preprocess_image(original_image)
            
            # 2. Predecir
            prob_media, incertidumbre = mc_dropout_predict(model, img_tensor)
            clase_predicha = "Neumonía" if prob_media > 0.5 else "Normal"
            confianza = prob_media if prob_media > 0.5 else (1 - prob_media)
            
            # 3. Grad-CAM
            gradcam_img = generate_gradcam(model, img_tensor, original_image)
            
            with col2:
                st.subheader(f"Mapa de Activación (Grad-CAM)")
                st.image(gradcam_img, use_container_width=True)
        
        # --- SECCIÓN DE RESULTADOS ---
        st.markdown("---")
        st.header("📊 Resultados Clínicos")
        
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("Predicción del Modelo", clase_predicha)
        res_col2.metric("Confianza", f"{confianza*100:.1f}%")
        # Mostrar incertidumbre en rojo si es alta
        color_incert = "normal" if incertidumbre < 0.05 else "inverse"
        res_col3.metric("Incertidumbre (Error ±)", f"{incertidumbre*100:.1f}%", delta_color=color_incert)

        # --- REPORTE CON LLM ---
        if gemini_key:
            st.markdown("---")
            st.header("📝 Reporte Autogenerado")
            with st.spinner('Redactando informe médico...'):
                try:
                    genai.configure(api_key=gemini_key)
                    llm_model = genai.GenerativeModel('gemini-2.5-flash')
                    prompt = f"""
                    Actúa como un radiólogo experto.
                    Resultados IA: Predicción: {clase_predicha}, Confianza: {confianza*100:.1f}%, Incertidumbre: ±{incertidumbre*100:.1f}%.
                    Redacta un breve informe clínico de máximo 3 párrafos.
                    Si la incertidumbre es alta (>5%) o la confianza es baja, emite una ALERTA CLÍNICA requiriendo revisión manual.
                    """
                    response = llm_model.generate_content(prompt)
                    st.write(response.text)
                except Exception as e:
                    st.error(f"Error al generar reporte con Gemini: {e}")
        else:
            st.warning("⚠️ Ingresa tu API Key de Gemini en la barra lateral para generar el reporte médico automatizado.")