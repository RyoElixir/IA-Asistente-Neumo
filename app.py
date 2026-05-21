import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import google.generativeai as genai

# --- 1. CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Asistente Radiológico IA", page_icon="🫁", layout="wide")

st.title("🫁 Asistente de Diagnóstico por IA: Radiografía de Tórax")
st.markdown("Sube una radiografía frontal de tórax para obtener un pre-diagnóstico con análisis de incertidumbre y explicabilidad visual.")

# --- 2. DEFINICIÓN DEL MODELO (Igual que en Kaggle) ---
class DenseNet121_MCDropout(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(DenseNet121_MCDropout, self).__init__()
        self.densenet = models.densenet121(weights=None)
        in_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        # Reescribimos el paso hacia adelante para evitar el "inplace=True"
        features = self.densenet.features(x)
        out = torch.relu(features) # <-- Aplicamos un ReLU seguro
        out = torch.nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.densenet.classifier(out)
        return out
        
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
    model.eval()
    
    features_blob = []
    gradients_blob = []
    
    # 1. Enganchar un recolector de matemáticas directamente sobre el tensor
    def hook_feature(module, input, output):
        features_blob.append(output)
        output.register_hook(lambda grad: gradients_blob.append(grad))
        
    target_layer = model.densenet.features
    handle = target_layer.register_forward_hook(hook_feature)
    
    # 2. Hacer que la IA evalúe la imagen
    output = model(image_tensor)
    pred_score = output[0, 0]
    
    # 3. Calcular los gradientes (El "por qué" tomó la decisión)
    model.zero_grad()
    pred_score.backward(retain_graph=True)
    handle.remove() # Limpiamos el gancho para no saturar la memoria
    
    # 4. Extraer los números
    activations = features_blob[0]
    gradients = gradients_blob[0]
    
    # 5. Ecuación de Grad-CAM
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]
        
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = torch.nn.functional.relu(heatmap)
    
    # Normalizar evitando división por cero
    max_val = torch.max(heatmap)
    if max_val > 0:
        heatmap /= max_val
        
    heatmap = heatmap.detach().cpu().numpy()

    # 6. Colorear y superponer sobre la radiografía
    img_resized = original_image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0

    heatmap_img = Image.fromarray(np.uint8(255 * heatmap))
    heatmap_resized = heatmap_img.resize((224, 224), Image.Resampling.LANCZOS)
    heatmap_resized = np.array(heatmap_resized) / 255.0

    cmap = plt.get_cmap('jet')
    heatmap_colored = cmap(heatmap_resized)[..., :3]

    alpha = 0.5
    overlay = heatmap_colored * alpha + img_array * (1 - alpha)
    overlay = np.clip(overlay, 0, 1)

    return overlay

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
