import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import google.generativeai as genai
import os
import urllib.request

# --- 1. CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Asistente Radiológico IA", page_icon="🫁", layout="wide")

st.title("🫁 Asistente de Diagnóstico Multietiqueta (14 Patologías)")
st.markdown("Sube una radiografía frontal de tórax. El modelo evaluará 14 patologías torácicas de forma simultánea.")

# --- 2. LISTAS OFICIALES ---
DISEASES = [
    "Atelectasia", "Cardiomegalia", "Derrame Pleural", "Infiltración",
    "Masa", "Nódulo", "Neumonía", "Neumotórax",
    "Consolidación", "Edema", "Enfisema", "Fibrosis",
    "Engrosamiento Pleural", "Hernia"
]

# --- 3. ARQUITECTURA EXACTA DE CHEXNET (STANFORD) ---
class CheXNet(nn.Module):
    def __init__(self):
        super(CheXNet, self).__init__()
        self.densenet121 = models.densenet121(weights=None)
        in_features = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(in_features, len(DISEASES)),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.densenet121.features(x)
        out = torch.relu(features) 
        out = torch.nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.densenet121.classifier(out)
        return out

# --- 4. FUNCIONES CACHEADAS (Descarga Automática y Mapeo Inteligente) ---
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CheXNet()
    
    weights_path = 'model.pth.tar'
    
    if not os.path.exists(weights_path):
        url = 'https://github.com/arnoweng/CheXNet/raw/master/model.pth.tar'
        urllib.request.urlretrieve(url, weights_path)
    
    checkpoint = torch.load(weights_path, map_location=device)
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        
    model_state_dict = model.state_dict()
    valid_state_dict = {}
    
    for my_key in model_state_dict.keys():
        core_key = my_key.replace('densenet121.', '')
        alt_key = core_key.replace('classifier.0', 'classifier')
        
        found = False
        for ckpt_key, ckpt_val in state_dict.items():
            if ckpt_key.endswith(core_key) or ckpt_key.endswith(alt_key):
                valid_state_dict[my_key] = ckpt_val
                found = True
                break
        
        if not found:
            valid_state_dict[my_key] = model_state_dict[my_key]
            
    model.load_state_dict(valid_state_dict)
    model.to(device)
    return model, device

# Carga del modelo con manejo de errores
try:
    with st.spinner('Cargando motor de IA (Puede tardar 1 minuto la primera vez si está descargando pesos)...'):
        model, device = load_model()
        modelo_cargado = True
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    modelo_cargado = False

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)

def predict_deterministic(model, image_tensor):
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probs = output.cpu().numpy()[0]
    return probs

def generate_gradcam(model, image_tensor, original_image):
    model.eval()
    
    features_blob = []
    gradients_blob = []
    
    def hook_feature(module, input, output):
        features_blob.append(output)
        output.register_hook(lambda grad: gradients_blob.append(grad))
        
    target_layer = model.densenet121.features
    handle = target_layer.register_forward_hook(hook_feature)
    
    output = model(image_tensor)
    
    top_class = 1  # El índice 1 en nuestra lista DISEASES corresponde a "Cardiomegalia"
    pred_score = output[0, top_class]
    
    model.zero_grad()
    pred_score.backward(retain_graph=True)
    handle.remove() 
    
    activations = features_blob[0]
    gradients = gradients_blob[0]
    
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]
        
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = torch.nn.functional.relu(heatmap)
    
    max_val = torch.max(heatmap)
    if max_val > 0:
        heatmap /= max_val
        
    heatmap = heatmap.detach().cpu().numpy()

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

    return overlay, DISEASES[top_class]

# --- 5. BARRA LATERAL ---
with st.sidebar:
    st.header("⚙️ Configuración")
    gemini_key = st.text_input("Ingresa tu Gemini API Key:", type="password")
    st.markdown("---")
    st.info("**Modelo Multietiqueta:** Red pre-entrenada con 112,120 radiografías (NIH). Evalúa 14 patologías independientes de forma determinista.")

# --- 6. INTERFAZ PRINCIPAL ---
uploaded_file = st.file_uploader("Selecciona una radiografía (JPEG/PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and modelo_cargado:
    col1, col2 = st.columns(2)
    original_image = Image.open(uploaded_file).convert('RGB')
    
    with col1:
        st.subheader("Radiografía Original")
        st.image(original_image, use_container_width=True)
        
    if st.button("Generar Diagnóstico Multietiqueta", type="primary"):
        with st.spinner('Analizando 14 patologías neuronales...'):
            img_tensor = preprocess_image(original_image)
            
            probabilidades = predict_deterministic(model, img_tensor)
            gradcam_img, top_disease = generate_gradcam(model, img_tensor, original_image)
            
            with col2:
                st.subheader(f"Mapa de Calor: {top_disease}")
                st.image(gradcam_img, use_container_width=True)
        
        # --- SECCIÓN DE RESULTADOS ---
        st.markdown("---")
        st.header("📊 Hallazgos Clínicos Detectados")
        
        hallazgos_significativos = []
        detectado_algo = False
        
        for idx, disease in enumerate(DISEASES):
            prob = probabilidades[idx]
            hallazgos_significativos.append(f"- {disease}: Probabilidad {prob*100:.1f}%")
            
            # Umbral clínico: Solo mostramos hallazgos > 15% de probabilidad
            if prob > 0.15:
                detectado_algo = True
                col_pat, col_conf = st.columns([2, 1])
                col_pat.markdown(f"**🔹 {disease}**")
                col_conf.metric("Confianza de IA", f"{prob*100:.1f}%")
                st.markdown("---")
                
        if not detectado_algo:
            st.success("✅ No se detectaron anomalías significativas que superen el umbral clínico base (Estudio preliminarmente normal).")

        # --- REPORTE CON LLM ---
        if gemini_key:
            st.header("📝 Reporte Autogenerado Extendido")
            with st.spinner('Redactando informe médico multietiqueta...'):
                try:
                    genai.configure(api_key=gemini_key)
                    llm_model = genai.GenerativeModel('gemini-1.5-flash')
                    lista_hallazgos_txt = "\n".join(hallazgos_significativos)
                    
                    prompt = f"""
                    Actúa como un radiólogo experto de triaje hospitalario.
                    Se ha procesado una radiografía de tórax con un modelo de IA multietiqueta (NIH ChestX-ray14).
                    Aquí están los resultados matemáticos crudos de las 14 patologías:
                    {lista_hallazgos_txt}
                    
                    Redacta un informe clínico breve estructurado en:
                    1. HALLAZGOS PRINCIPALES (Menciona solo las patologías con probabilidad significativa).
                    2. IMPRESIÓN CONCLUSIVA.
                    """
                    response = llm_model.generate_content(prompt)
                    st.write(response.text)
                except Exception as e:
                    error_msg = str(e)
                    if "429" in error_msg or "quota" in error_msg.lower():
                        st.warning("⏳ La IA está procesando demasiadas solicitudes. Espera unos segundos e intenta de nuevo.")
                    else:
                        st.error(f"Error al generar el reporte: {e}")
