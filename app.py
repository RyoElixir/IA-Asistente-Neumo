import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import google.generativeai as genai

# --- 1. CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Asistente Radiológico IA", page_icon="🫁", layout="wide")

st.title("🫁 Asistente de Diagnóstico Multietiqueta (14 Patologías)")
st.markdown("Sube una radiografía frontal de tórax. El modelo evaluará 14 patologías torácicas de forma simultánea e independiente.")

# --- 2. LISTA OFICIAL DE PATOLOGÍAS (NIH ChestX-ray14) ---
DISEASES = [
    "Atelectasia", "Cardiomegalia", "Derrame Pleural", "Infiltración",
    "Masa", "Nódulo", "Neumonía", "Neumotórax",
    "Consolidación", "Edema", "Enfisema", "Fibrosis",
    "Engrosamiento Pleural", "Hernia"
]

# --- 3. DEFINICIÓN DEL MODELO (Multietiqueta y Seguro para Hooks) ---
class DenseNet121_MultiLabel(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(DenseNet121_MultiLabel, self).__init__()
        self.densenet = models.densenet121(weights=None)
        in_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, len(DISEASES)) # 14 Salidas
        )

    def forward(self, x):
        # Reescribimos el forward para evitar el inplace=True de ReLU y que el Grad-CAM no falle
        features = self.densenet.features(x)
        out = torch.relu(features) 
        out = torch.nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.densenet.classifier(out)
        return out
        
    def enable_dropout(self):
        for m in self.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

# --- 4. FUNCIONES CACHEADAS ---
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DenseNet121_MultiLabel()
    # AQUÍ CARGAREMOS LOS PESOS DE LAS 14 ENFERMEDADES
    model.load_state_dict(torch.load('chexnet_14_weights.pth', map_location=device))
    model.to(device)
    return model, device

# Usamos try-except por si aún no has subido el nuevo archivo de pesos
try:
    model, device = load_model()
    modelo_cargado = True
except FileNotFoundError:
    st.error("⚠️ Falta el archivo 'chexnet_14_weights.pth'. Por favor súbelo a tu repositorio.")
    modelo_cargado = False

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
    all_passes_preds = []
    
    with torch.no_grad():
        for _ in range(num_passes):
            output = model(image_tensor)
            # Aplicamos Sigmoid a las 14 salidas simultáneamente
            probs = torch.sigmoid(output).cpu().numpy()[0]
            all_passes_preds.append(probs)
            
    mean_preds = np.mean(all_passes_preds, axis=0)
    std_preds = np.std(all_passes_preds, axis=0)
    return mean_preds, std_preds

def generate_gradcam(model, image_tensor, original_image):
    model.eval()
    features_blob = []
    gradients_blob = []
    
    # Enganchar el recolector matemáticamente
    def hook_feature(module, input, output):
        features_blob.append(output)
        output.register_hook(lambda grad: gradients_blob.append(grad))
        
    target_layer = model.densenet.features
    handle = target_layer.register_forward_hook(hook_feature)
    
    output = model(image_tensor)
    
    # Extraer el mapa de la enfermedad con MAYOR probabilidad detectada
    top_class = torch.argmax(output).item()
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

    # Colorear sin usar CV2
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
    st.info("**Modelo Multietiqueta:** Basado en la arquitectura CheXNet evaluando 14 patologías simultáneas.")

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
            prob_medias, incertidumbres = mc_dropout_predict(model, img_tensor)
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
            prob = prob_medias[idx]
            incert = incertidumbres[idx]
            hallazgos_significativos.append(f"- {disease}: Probabilidad {prob*100:.1f}%, Incertidumbre ±{incert*100:.1f}%")
            
            # Solo mostramos en pantalla los hallazgos con más de un 15% de probabilidad
            if prob > 0.15:
                detectado_algo = True
                col_pat, col_conf, col_inc = st.columns(3)
                col_pat.markdown(f"**🔹 {disease}**")
                col_conf.metric("Confianza", f"{prob*100:.1f}%")
                col_inc.metric("Incertidumbre", f"{incert*100:.1f}%", delta_color="inverse" if incert > 0.05 else "normal")
                st.markdown("---")
                
        if not detectado_algo:
            st.success("✅ No se detectaron anomalías significativas que superen el umbral clínico base (Estudio preliminarmente normal).")

        # --- REPORTE CON LLM ---
        if gemini_key:
            st.header("📝 Reporte Autogenerado Extendido")
            with st.spinner('Redactando informe médico multietiqueta...'):
                try:
                    genai.configure(api_key=gemini_key)
                    # Cambiamos a gemini-1.5-flash para tener una cuota gratuita más holgada
                    llm_model = genai.GenerativeModel('gemini-1.5-flash')
                    lista_hallazgos_txt = "\n".join(hallazgos_significativos)
                    
                    prompt = f"""
                    Actúa como un radiólogo experto de triaje hospitalario.
                    Se ha procesado una radiografía de tórax con un modelo de IA multietiqueta (NIH ChestX-ray14).
                    Aquí están los resultados matemáticos crudos de las 14 patologías:
                    {lista_hallazgos_txt}
                    
                    Redacta un informe clínico breve estructurado en:
                    1. HALLAZGOS PRINCIPALES (Menciona solo las patologías con probabilidad significativa y su incertidumbre).
                    2. IMPRESIÓN CONCLUSIVA (Si hay alta incertidumbre o peligro, emite una ALERTA CLÍNICA).
                    """
                    response = llm_model.generate_content(prompt)
                    st.write(response.text)
                except Exception as e:
                    error_msg = str(e)
                    if "429" in error_msg or "quota" in error_msg.lower():
                        st.warning("⏳ La IA está procesando demasiadas solicitudes (Límite de cuota gratuita de Google). Espera unos segundos y vuelve a presionar el botón.")
                    else:
                        st.error(f"Ocurrió un error al contactar a la IA: {e}")
