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

st.title("🫁 Asistente de Diagnóstico Multietiqueta (Calibrado)")
st.markdown("Sube una radiografía frontal de tórax. El modelo evaluará 14 patologías usando **umbrales dinámicos calibrados** para mitigar el sesgo de datos.")

# --- 2. LISTAS Y UMBRALES (INGENIERÍA DE DATOS) ---
DISEASES = [
    "Atelectasia", "Cardiomegalia", "Derrame Pleural", "Infiltración",
    "Masa", "Nódulo", "Neumonía", "Neumotórax",
    "Consolidación", "Edema", "Enfisema", "Fibrosis",
    "Engrosamiento Pleural", "Hernia"
]

# Umbrales personalizados: Castigamos las mayorías (Infiltración) y somos muy sensibles con las minorías (Nódulo/Masa)
THRESHOLDS = [
    0.15,  # Atelectasia
    0.10,  # Cardiomegalia (Sensible)
    0.15,  # Derrame Pleural
    0.40,  # Infiltración (Muy estricto, evita falsos positivos)
    0.08,  # Masa (Alta sensibilidad)
    0.05,  # Nódulo (Alta sensibilidad)
    0.20,  # Neumonía
    0.05,  # Neumotórax (Alta sensibilidad)
    0.15,  # Consolidación
    0.15,  # Edema
    0.05,  # Enfisema
    0.10,  # Fibrosis
    0.10,  # Engrosamiento Pleural
    0.02   # Hernia
]

# --- 3. ARQUITECTURA CHEXNET ---
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

# --- 4. FUNCIONES CACHEADAS ---
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CheXNet()
    
    weights_path = 'model.pth.tar'
    if not os.path.exists(weights_path):
        url = 'https://github.com/arnoweng/CheXNet/raw/master/model.pth.tar'
        urllib.request.urlretrieve(url, weights_path)
    
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        
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

try:
    with st.spinner('Cargando motor de IA...'):
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
    probs = output[0].detach().cpu().numpy()
    
    # MAGIA MATEMÁTICA: Elegir la enfermedad por "Gravedad Relativa"
    umbrales_array = np.array(THRESHOLDS)
    gravedad_relativa = probs / umbrales_array
    top_class = np.argmax(gravedad_relativa) # Elegimos la que más superó su propio límite
    
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
    st.info("**Calibración Activa:** Los resultados son filtrados mediante matriz de umbrales para evitar falsos positivos por desbalance de clases.")

# --- 6. INTERFAZ PRINCIPAL ---
uploaded_file = st.file_uploader("Selecciona una radiografía (JPEG/PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and modelo_cargado:
    col1, col2 = st.columns(2)
    original_image = Image.open(uploaded_file).convert('RGB')
    
    with col1:
        st.subheader("Radiografía Original")
        st.image(original_image, use_container_width=True)
        
    if st.button("Generar Diagnóstico Calibrado", type="primary"):
        with st.spinner('Aplicando algoritmos de gravedad relativa...'):
            img_tensor = preprocess_image(original_image)
            
            probabilidades = predict_deterministic(model, img_tensor)
            gradcam_img, top_disease = generate_gradcam(model, img_tensor, original_image)
            
            with col2:
                st.subheader(f"Mapa de Calor: {top_disease}")
                st.image(gradcam_img, use_container_width=True)
        
        # --- SECCIÓN DE RESULTADOS ---
        st.markdown("---")
        st.header("📊 Hallazgos Clínicos (Filtrados por Umbral)")
        
        hallazgos_significativos = []
        detectado_algo = False
        
        for idx, disease in enumerate(DISEASES):
            prob = probabilidades[idx]
            umbral = THRESHOLDS[idx]
            
            # Solo mostramos si la probabilidad VENCE a su propio umbral
            if prob >= umbral:
                detectado_algo = True
                hallazgos_significativos.append(f"- {disease}: Probabilidad {prob*100:.1f}%")
                
                col_pat, col_conf = st.columns([2, 1])
                col_pat.markdown(f"**🔹 {disease}** *(Umbral exigido: {umbral*100:.0f}%)*")
                col_conf.metric("Confianza de IA", f"{prob*100:.1f}%")
                st.markdown("---")
                
        if not detectado_algo:
            st.success("✅ Estudio preliminarmente normal. Ninguna métrica superó los umbrales de alerta clínica.")
            hallazgos_significativos.append("Estudio normal. No se superaron los umbrales de riesgo.")

        # --- REPORTE CON LLM ---
        if gemini_key:
            st.header("📝 Reporte Autogenerado Extendido")
            with st.spinner('Redactando informe médico...'):
                try:
                    genai.configure(api_key=gemini_key)
                    # VOLVEMOS AL MODELO 2.5 QUE NO DA ERROR 404
                    llm_model = genai.GenerativeModel('gemini-2.5-flash')
                    lista_hallazgos_txt = "\n".join(hallazgos_significativos)
                    
                    prompt = f"""
                    Actúa como un radiólogo experto de triaje hospitalario.
                    Resultados post-calibración de IA (solo se muestran los que superaron el umbral clínico de riesgo):
                    {lista_hallazgos_txt}
                    
                    Redacta un informe breve:
                    1. HALLAZGOS PRINCIPALES.
                    2. IMPRESIÓN CONCLUSIVA.
                    """
                    response = llm_model.generate_content(prompt)
                    st.write(response.text)
                except Exception as e:
                    error_msg = str(e)
                    if "429" in error_msg or "quota" in error_msg.lower():
                        st.warning("⏳ La IA está procesando demasiadas solicitudes. Espera unos 30 segundos e intenta de nuevo.")
                    else:
                        st.error(f"Error al generar el reporte: {e}")
