import streamlit as st
import os
import shutil
import glob
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from streamlit_drawable_canvas import st_canvas

# >>> DEPENDÊNCIA DE TKINTER REMOVIDA PARA EVITAR 'ImportError: libtk8.6.so' <<<

# --- 1. Inicialização do Session State (Robusto) ---

def initialize_session_state():
    """Garante que todas as chaves necessárias existam."""
    if 'DATASET_ROOT' not in st.session_state:
        st.session_state.DATASET_ROOT = None
    if 'IMAGE_FILES' not in st.session_state:
        st.session_state.IMAGE_FILES = []
    if 'CURRENT_IMAGE_INDEX' not in st.session_state:
        st.session_state.CURRENT_IMAGE_INDEX = 0
    if 'BOUNDING_BOXES' not in st.session_state:
        st.session_state.BOUNDING_BOXES = [] 
    if 'show_division' not in st.session_state:
        st.session_state.show_division = False
    if 'current_class_id' not in st.session_state:
        st.session_state.current_class_id = 0
    if 'LAST_CLASS_ID' not in st.session_state:
        st.session_state.LAST_CLASS_ID = 0

# --- 2. Funções de Ajuda e Lógica de Labeling ---

def to_yolo_format(box_data, img_width, img_height, class_id):
    """Converte coordenadas do canvas para o formato YOLO normalizado."""
    
    x = box_data['left']
    y = box_data['top']
    w = box_data['width']
    h = box_data['height']
    
    # Conversão para YOLO (x_center, y_center, w_norm, h_norm)
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    w_norm = w / img_width
    h_norm = h / img_height
    
    return f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"

def save_labels(image_path, boxes_data):
    """
    Cria ou atualiza o arquivo .txt na pasta 'labels' (Requisitos 5, 6, 7).
    """
    if not boxes_data:
        st.warning("Nenhum bounding box para salvar.")
        return False

    img = Image.open(image_path)
    img_width, img_height = img.size
    
    # Requisito 5: Cria a pasta 'labels' no mesmo diretório pai das imagens
    image_dir = os.path.dirname(image_path)
    label_dir = os.path.join(os.path.dirname(image_dir), 'labels') 
    
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    # Requisito 6: Cria o nome do arquivo .txt
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    label_file_path = os.path.join(label_dir, f"{base_name}.txt")

    yolo_lines = []
    
    for box_info in boxes_data:
        class_id = box_info.get('class_id', st.session_state.LAST_CLASS_ID)
        yolo_line = to_yolo_format(box_info, img_width, img_height, class_id)
        yolo_lines.append(yolo_line)

    with open(label_file_path, 'w') as f:
        f.write('\n'.join(yolo_lines))
        
    st.session_state.BOUNDING_BOXES = [] 
    return True

def split_and_move_dataset(X_paths, Y_paths, root_dir, train_r, test_r, valid_r):
    """
    Divide o dataset e move os arquivos de Imagem e Label para a estrutura de pastas do YOLO.
    (Requisitos 9, 11, 12)
    """
    if train_r + test_r + valid_r != 100:
        st.error("A soma das proporções deve ser 100%.")
        return

    st.info("Iniciando a divisão e movimentação dos arquivos...")
    
    train_size = train_r / 100.0
    test_size = test_r / 100.0
    
    indices = np.arange(len(X_paths))
    train_indices, temp_indices = train_test_split(
        indices, test_size=(1 - train_size), random_state=42
    )
    
    if 1 - train_size > 0:
        test_ratio_adj = test_size / (1 - train_size)
    else:
        test_ratio_adj = 0 

    test_indices, valid_indices = train_test_split(
        temp_indices, test_size=(1 - test_ratio_adj), random_state=42
    )

    datasets = {
        'train': train_indices,
        'test': test_indices,
        'valid': valid_indices,
    }

    output_dir = os.path.join(os.path.dirname(root_dir), "YOLO_Dataset_Split")
    if os.path.exists(output_dir):
         st.warning("Pasta de saída já existe. Arquivos serão sobrescritos.")
    else:
         os.makedirs(output_dir)
        
    for name, indices in datasets.items():
        img_target_dir = os.path.join(output_dir, name, 'images')
        lbl_target_dir = os.path.join(output_dir, name, 'labels')
        
        os.makedirs(img_target_dir, exist_ok=True)
        os.makedirs(lbl_target_dir, exist_ok=True)
        
        st.write(f"Movendo {len(indices)} arquivos para **{name}**...")
        
        for i in indices:
            shutil.copy(X_paths[i], img_target_dir)
            if os.path.exists(Y_paths[i]):
                 shutil.copy(Y_paths[i], lbl_target_dir)

    st.success(f"Divisão concluída! O dataset foi salvo em: **{output_dir}**")
    st.info("As pastas 'train', 'test' e 'valid' foram criadas neste diretório.")
    st.session_state.show_division = False

def next_image():
    """Avança para a próxima imagem."""
    if st.session_state.CURRENT_IMAGE_INDEX < len(st.session_state.IMAGE_FILES) - 1:
        st.session_state.CURRENT_IMAGE_INDEX += 1
    else:
        st.info("Fim da lista de imagens.")

def prev_image():
    """Volta para a imagem anterior."""
    if st.session_state.CURRENT_IMAGE_INDEX > 0:
        st.session_state.CURRENT_IMAGE_INDEX -= 1
    else:
        st.info("Primeira imagem da lista.")

def load_images_from_path(path_to_process):
    """Lógica centralizada para carregar imagens e atualizar o session state."""
    st.session_state.DATASET_ROOT = path_to_process
    
    all_files = glob.glob(os.path.join(path_to_process, "*.*"))
    img_extensions = ['.jpg', '.jpeg', '.png']
    st.session_state.IMAGE_FILES = [f for f in all_files if os.path.splitext(f)[1].lower() in img_extensions]
    st.session_state.CURRENT_IMAGE_INDEX = 0
    st.session_state.BOUNDING_BOXES = []
    st.success(f"Encontradas {len(st.session_state.IMAGE_FILES)} imagens em: {path_to_process}")
    
# --- 3. Execução Principal ---

initialize_session_state()
st.set_page_config(layout="wide", page_title="Labeling Tool (YOLO Format)")
st.title("🎯 Plataforma de Anotação de Bounding Box (Streamlit/YOLO)")

# --- BARRA LATERAL (Controles e Navegação) ---
with st.sidebar:
    st.header("⚙️ Controles do Dataset")
    
    st.subheader("1. Carregar Pasta de Imagens (Requisito 1)")
    
    # Input de texto para o caminho da pasta
    temp_root = st.text_input(
        "Insira o caminho absoluto da pasta de imagens:", 
        value=st.session_state.DATASET_ROOT or "",
        key='manual_path_input',
        help="Ex: /caminho/para/pasta/images. Copie e cole aqui."
    )
    
    if st.button("Carregar Pasta", type="primary"):
        if os.path.isdir(temp_root):
            load_images_from_path(temp_root)
            st.experimental_rerun()
        else:
            st.error("Caminho inválido ou pasta não encontrada.")

    # Exibe o caminho atual
    if st.session_state.DATASET_ROOT:
         st.info(f"Pasta de trabalho: **{st.session_state.DATASET_ROOT}**")
    
    st.markdown("---") 

    # Requisito 2: Lista de Imagens Clicável e Navegação
    if st.session_state.IMAGE_FILES:
        st.subheader("2. Navegação e Status")
        
        # Navegação
        col_prev, col_next = st.columns(2)
        col_prev.button("⬅️ Anterior", on_click=prev_image)
        col_next.button("Próxima ➡️", on_click=next_image)
        
        st.markdown("---")
        
        current_idx = st.session_state.CURRENT_IMAGE_INDEX
        img_name = os.path.basename(st.session_state.IMAGE_FILES[current_idx])
        st.info(f"Visualizando: **{img_name}** ({current_idx + 1}/{len(st.session_state.IMAGE_FILES)})")

        # Configuração da Classe (Requisito 4)
        st.subheader("3. Classe de Marcação")
        st.session_state.current_class_id = st.number_input(
            "ID da Classe (Inteiro):", 
            min_value=0, 
            value=st.session_state.LAST_CLASS_ID, 
            step=1,
            key='class_input_box'
        )
        st.session_state.LAST_CLASS_ID = st.session_state.current_class_id
        st.caption(f"A próxima caixa desenhada será rotulada como CLASSE: **{st.session_state.current_class_id}**")
        
# --- ÁREA PRINCIPAL (Visualização e Anotação) ---

if st.session_state.IMAGE_FILES:
    current_image_path = st.session_state.IMAGE_FILES[st.session_state.CURRENT_IMAGE_INDEX]
    
    try:
        img = Image.open(current_image_path).convert("RGB")
        img_width, img_height = img.size
        
        # Requisito 3: Canvas Desenho
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=2,
            stroke_color="#FF0000",
            background_image=img,
            update_streamlit=True,
            height=min(img_height, 700),
            width=min(img_width, 900),
            drawing_mode="rect", # Requisito 3: Modo Retângulo
            key="canvas_draw"
        )
        
        st.markdown("---")
        
        # Processamento dos boxes desenhados (Requisito 4/7)
        if canvas_result.json_data is not None and canvas_result.json_data['objects']:
            st.subheader("Boxes Desenhados e Prontos para Salvar")
            
            final_boxes_to_save = []
            
            for i, obj in enumerate(canvas_result.json_data['objects']):
                if obj['type'] == 'rect':
                    box_info = {
                        'left': obj['left'],
                        'top': obj['top'],
                        'width': obj['width'],
                        'height': obj['height'],
                        'class_id': st.session_state.current_class_id
                    }
                    final_boxes_to_save.append(box_info)
                    st.write(f"Box {i+1} - Classe: **{st.session_state.current_class_id}**")
            
            # Botão de salvar (Requisito 8: Concluído)
            if st.button("💾 Salvar Labels e Avançar (Requisito 8)", type="success"):
                if save_labels(current_image_path, final_boxes_to_save):
                    st.success(f"Labels salvos para {os.path.basename(current_image_path)} e avançando!")
                    next_image()
                    st.experimental_rerun()
                else:
                    st.error("Erro ao salvar labels.")


    except Exception as e:
        st.error(f"Erro ao carregar ou exibir imagem: {e}")
        st.warning("Verifique se o caminho da pasta está correto e se o arquivo existe.")


# --- 4. Divisão do Dataset (Requisitos 10, 11, 12) ---

st.sidebar.markdown("---")
if st.sidebar.button("📊 Abrir Ferramenta de Divisão (Requisito 10)"):
    if not st.session_state.IMAGE_FILES:
        st.sidebar.warning("Carregue uma pasta de imagens primeiro.")
    else:
        st.session_state.show_division = True

if st.session_state.get('show_division', False):
    st.header("Divisão do Dataset (Treino/Teste/Validação)")
    
    # Requisito 11: Sliders de Proporção
    col1, col2, col3 = st.columns(3)
    train_ratio = col1.slider("Treino (%)", 0, 100, 70, key="train_r")
    test_ratio = col2.slider("Teste (%)", 0, 100, 20, key="test_r")
    valid_ratio = col3.slider("Validação (%)", 0, 100, 10, key="valid_r")
    
    if train_ratio + test_ratio + valid_ratio != 100:
        st.error("A soma das proporções deve ser 100%. Por favor, ajuste os sliders.")
    else:
        image_list = st.session_state.IMAGE_FILES
        
        image_dir = st.session_state.DATASET_ROOT
        label_dir = os.path.join(os.path.dirname(image_dir), 'labels')
        
        label_list = [
            os.path.join(label_dir, f"{os.path.splitext(os.path.basename(f))[0]}.txt") 
            for f in image_list
        ]
        
        existing_pairs = [(img, lbl) for img, lbl in zip(image_list, label_list) if os.path.exists(lbl)]
        
        if not existing_pairs:
            st.warning("Nenhum par Imagem/Label (.txt) encontrado. Rotule as imagens e clique em Salvar primeiro!")
        else:
            X = [pair[0] for pair in existing_pairs]
            Y = [pair[1] for pair in existing_pairs]
            st.info(f"Pronto para dividir {len(X)} pares de Imagem/Label.")
            
            if st.button("Executar Divisão (Requisito 12)", type="secondary"):
                split_and_move_dataset(
                    X, Y, 
                    st.session_state.DATASET_ROOT, 
                    train_ratio, test_ratio, valid_ratio
                )
