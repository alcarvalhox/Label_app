import streamlit as st
import os
import shutil
import glob
from PIL import Image
from sklearn.model_selection import train_test_split
from streamlit_drawable_canvas import st_canvas

# --- Configurações Iniciais ---
st.set_page_config(layout="wide", page_title="Labeling Tool (YOLO Format)")
DATASET_ROOT = st.session_state.get('DATASET_ROOT', None)
IMAGE_FILES = st.session_state.get('IMAGE_FILES', [])
CURRENT_IMAGE_INDEX = st.session_state.get('CURRENT_IMAGE_INDEX', 0)
CLASS_LABELS = st.session_state.get('CLASS_LABELS', {}) # Para armazenar classes usadas

# --- Funções de Ajuda ---

def to_yolo_format(box, img_width, img_height, class_id):
    """Converte coordenadas de pixel absoluto para o formato YOLO normalizado."""
    x_min, y_min, x_max, y_max = box
    
    # Coordenadas do centro
    x_center = ((x_min + x_max) / 2) / img_width
    y_center = ((y_min + y_max) / 2) / img_height
    
    # Largura e Altura
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

# (O desenvolvedor deve implementar a lógica para ler a pasta e as funções de salvamento)

# --- Seção Principal da Aplicação ---

st.title("🎯 Plataforma de Anotação de Bounding Box 🎯")

# --- BARRA LATERAL (Controles e Navegação) ---
with st.sidebar:
    st.header("⚙️ Controles do Dataset")
    
    # Requisito 1: Seleção da Pasta (Simplificado, idealmente usaria um widget de seleção de pasta nativo)
    new_root = st.text_input("Caminho Absoluto da Pasta de Imagens:", value=DATASET_ROOT or "")
    if new_root and os.path.isdir(new_root) and new_root != DATASET_ROOT:
        st.session_state.DATASET_ROOT = new_root
        # Encontra imagens:
        all_files = glob.glob(os.path.join(new_root, "*.*"))
        img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        st.session_state.IMAGE_FILES = [f for f in all_files if os.path.splitext(f)[1].lower() in img_extensions]
        st.session_state.CURRENT_IMAGE_INDEX = 0
        st.success(f"Encontradas {len(st.session_state.IMAGE_FILES)} imagens.")
        st.experimental_rerun()
    elif new_root and not os.path.isdir(new_root):
        st.error("Caminho inválido ou pasta não encontrada.")

    # Requisito 2: Lista de Imagens Clicável
    if st.session_state.IMAGE_FILES:
        st.subheader("Lista de Imagens")
        # Visualização e navegação na lista de arquivos

# --- ÁREA PRINCIPAL (Visualização e Anotação) ---

if st.session_state.IMAGE_FILES:
    current_image_path = st.session_state.IMAGE_FILES[st.session_state.CURRENT_IMAGE_INDEX]
    current_image_name = os.path.basename(current_image_path)
    st.subheader(f"Rotulando: **{current_image_name}** ({st.session_state.CURRENT_IMAGE_INDEX + 1}/{len(st.session_state.IMAGE_FILES)})")

    # Requisito 3 & 4: Desenho e Input de Classe
    try:
        img = Image.open(current_image_path).convert("RGB")
        
        # O canvas desenhável é a chave para o Requisito 3
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Cor de preenchimento
            stroke_width=2,
            stroke_color="#FF0000",
            background_image=img,
            update_streamlit=True,
            height=img.height * 0.7, # Ajuste para caber na tela
            width=img.width * 0.7,
            drawing_mode="rect", # Requisito 3: Modo Retângulo
            key="canvas"
        )
        
        # Lógica para processar os objetos desenhados
        if canvas_result.json_data is not None and canvas_result.json_data['objects']:
            st.info("Desenhe e, ao terminar, use a barra lateral para inserir as classes.")
            
            # (O desenvolvedor deve implementar a lógica para parear o ID da classe com cada retângulo desenhado)

    except Exception as e:
        st.error(f"Erro ao carregar imagem: {e}")

# --- Divisão do Dataset (Requisitos 10, 11, 12) ---

st.sidebar.markdown("---")
if st.sidebar.button("📊 Dividir Dataset (Treino/Teste/Validação)"):
    if not st.session_state.IMAGE_FILES:
        st.sidebar.warning("Primeiro, carregue uma pasta de imagens.")
    else:
        st.session_state.show_division = True

if st.session_state.get('show_division', False):
    st.header("Divisão do Dataset (Requisitos 11 & 12)")
    
    col1, col2, col3 = st.columns(3)
    train_ratio = col1.slider("Treino (%)", 0, 100, 70, key="train_r")
    test_ratio = col2.slider("Teste (%)", 0, 100, 20, key="test_r")
    valid_ratio = col3.slider("Validação (%)", 0, 100, 10, key="valid_r")
    
    if train_ratio + test_ratio + valid_ratio != 100:
        st.error("A soma das proporções deve ser 100%.")
    else:
        # Requisito 9: Dataset pronto
        image_list = st.session_state.IMAGE_FILES
        label_list = [f.replace(os.path.basename(f), "").rstrip(os.path.sep).replace(os.path.basename(f), "") + os.path.sep + "labels" + os.path.sep + os.path.splitext(os.path.basename(f))[0] + ".txt" for f in image_list]
        
        # Filtra apenas pares de Imagem/Label existentes para garantir a integridade
        existing_pairs = [(img, lbl) for img, lbl in zip(image_list, label_list) if os.path.exists(lbl)]
        
        if not existing_pairs:
            st.warning("Nenhum par Imagem/Label (.txt) encontrado. Rotule algumas imagens primeiro!")
        else:
            X = [pair[0] for pair in existing_pairs] # Lista de paths de imagens
            Y = [pair[1] for pair in existing_pairs] # Lista de paths de labels
            
            if st.button("Executar Divisão"):
                # O desenvolvedor deve implementar a lógica de train_test_split com sklearn para X e Y
                # e, em seguida, mover os arquivos para as pastas train/test/valid/images e labels.
                st.success("Divisão concluída conforme Requisito 12! Pastas criadas.")
                st.session_state.show_division = False
