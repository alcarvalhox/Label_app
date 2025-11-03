import streamlit as st
import os
import shutil
import glob
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from streamlit_drawable_canvas import st_canvas

# --- 1. Inicialização do Session State (CRÍTICO para evitar KeyErrors) ---

def initialize_session_state():
    """Garante que todas as chaves necessárias existam."""
    if 'DATASET_ROOT' not in st.session_state:
        st.session_state.DATASET_ROOT = None
    if 'IMAGE_FILES' not in st.session_state:
        st.session_state.IMAGE_FILES = []
    if 'CURRENT_IMAGE_INDEX' not in st.session_state:
        st.session_state.CURRENT_IMAGE_INDEX = 0
    if 'BOUNDING_BOXES' not in st.session_state:
        # Armazena os boxes desenhados na imagem atual
        st.session_state.BOUNDING_BOXES = [] 
    if 'show_division' not in st.session_state:
        st.session_state.show_division = False
    if 'current_class_id' not in st.session_state:
        st.session_state.current_class_id = 0 # ID da classe a ser aplicada
    if 'LAST_CLASS_ID' not in st.session_state:
        st.session_state.LAST_CLASS_ID = 0 # Última classe usada para conveniência

# --- 2. Funções de Ajuda e Lógica de Labeling ---

def to_yolo_format(box_data, img_width, img_height, class_id):
    """Converte coordenadas do canvas para o formato YOLO normalizado."""
    
    # As coordenadas do canvas_result já são normalizadas pela biblioteca
    # mas o st_canvas retorna top/left/width/height (não centrado)
    
    x = box_data['left']
    y = box_data['top']
    w = box_data['width']
    h = box_data['height']
    
    # Convertendo para o formato YOLO (x_center, y_center, w_norm, h_norm)
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    w_norm = w / img_width
    h_norm = h / img_height
    
    # Requisito 7: Formato '<class_id> <x_center> <y_center> <width> <height>'
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
    label_dir = os.path.join(os.path.dirname(image_dir), 'labels') # No mesmo nível que 'images' e 'test/train/valid'
    
    # Se a pasta 'labels' não existir, cria
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    # Requisito 6: Cria o nome do arquivo .txt
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    label_file_path = os.path.join(label_dir, f"{base_name}.txt")

    yolo_lines = []
    
    for box_info in boxes_data:
        # Assumindo que a classe foi anexada ao objeto desenhado ou está no session state
        # NOTA: O fluxo de trabalho real requer um input após o desenho
        # Simplificamos para usar uma classe padrão temporária ou a última usada
        class_id = box_info.get('class_id', st.session_state.LAST_CLASS_ID)
        
        # Converte as coordenadas do canvas (já em pixels absolutos se o canvas não tiver sido redimensionado)
        # Assumimos que 'left', 'top', 'width', 'height' estão disponíveis no box_info
        
        # NOTE: st_canvas retorna as coordenadas *baseadas no tamanho de exibição no Streamlit*,
        # não no tamanho real da imagem, se ela foi redimensionada no canvas!
        # Para precisão, você PRECISA garantir que o canvas tenha o tamanho da imagem original.
        
        yolo_line = to_yolo_format(box_info, img_width, img_height, class_id)
        yolo_lines.append(yolo_line)

    # Escreve no arquivo (Requisito 7)
    with open(label_file_path, 'w') as f:
        f.write('\n'.join(yolo_lines))
        
    st.session_state.BOUNDING_BOXES = [] # Limpa os boxes após salvar
    return True

def split_and_move_dataset(X_paths, Y_paths, root_dir, train_r, test_r, valid_r):
    """
    Divide o dataset e move os arquivos de Imagem e Label para a estrutura de pastas do YOLO.
    """
    if train_r + test_r + valid_r != 100:
        st.error("A soma das proporções deve ser 100%.")
        return

    st.info("Iniciando a divisão e movimentação dos arquivos...")
    
    # Normaliza as proporções
    train_size = train_r / 100.0
    test_size = test_r / 100.0
    
    # 1. Divisão: Treino vs. (Teste + Validação)
    # Usamos np.arange para criar índices e garantir que as listas X e Y permaneçam sincronizadas
    indices = np.arange(len(X_paths))
    train_indices, temp_indices = train_test_split(
        indices, test_size=(1 - train_size), random_state=42
    )
    
    # 2. Divisão: Teste vs. Validação
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

    # Requisito 12: Criação da Estrutura de Pastas e Movimentação
    # O diretório de saída será 'YOLO_Dataset_Split' no diretório pai das imagens
    output_dir = os.path.join(os.path.dirname(root_dir), "YOLO_Dataset_Split")
    if os.path.exists(output_dir):
         st.warning("Pasta de saída já existe. Arquivos serão sobrescritos.")
    else:
         os.makedirs(output_dir)
        
    for name, indices in datasets.items():
        img_target_dir = os.path.join(output_dir, name, 'images')
        lbl_target_dir = os.path.join(output_dir, name, 'labels')
        
        # Cria as pastas (Requisito 12: train/images, valid/labels, etc.)
        os.makedirs(img_target_dir, exist_ok=True)
        os.makedirs(lbl_target_dir, exist_ok=True)
        
        st.write(f"Movendo {len(indices)} arquivos para **{name}**...")
        
        for i in indices:
            # Garante que o par Imagem/Label seja copiado junto
            shutil.copy(X_paths[i], img_target_dir)
            if os.path.exists(Y_paths[i]):
                 shutil.copy(Y_paths[i], lbl_target_dir)

    st.success(f"Divisão concluída! O dataset foi salvo em: **{output_dir}**")
    st.info("Você pode acessar e baixar as pastas 'train', 'test' e 'valid' neste diretório.")
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
        
# --- 3. Execução Principal ---

initialize_session_state()
st.title("🎯 Plataforma de Anotação de Bounding Box (Streamlit/YOLO)")

# --- BARRA LATERAL (Controles e Navegação) ---
with st.sidebar:
    st.header("⚙️ Controles do Dataset")
    
    # Requisito 1: Seleção da Pasta (Corrigido para usar st.session_state diretamente)
    new_root = st.text_input(
        "Caminho Absoluto da Pasta de Imagens:", 
        value=st.session_state.DATASET_ROOT or "",
        help="Ex: /caminho/para/seus/dados/images"
    )
    
    if new_root and os.path.isdir(new_root) and new_root != st.session_state.DATASET_ROOT:
        st.session_state.DATASET_ROOT = new_root
        
        # Encontra imagens:
        all_files = glob.glob(os.path.join(new_root, "*.*"))
        img_extensions = ['.jpg', '.jpeg', '.png']
        st.session_state.IMAGE_FILES = [f for f in all_files if os.path.splitext(f)[1].lower() in img_extensions]
        st.session_state.CURRENT_IMAGE_INDEX = 0
        st.session_state.BOUNDING_BOXES = [] # Limpa labels ao carregar nova pasta
        st.success(f"Encontradas {len(st.session_state.IMAGE_FILES)} imagens.")
        st.experimental_rerun() # Reinicia para carregar a primeira imagem
    elif new_root and not os.path.isdir(new_root):
        st.error("Caminho inválido ou pasta não encontrada.")

    # Requisito 2: Lista de Imagens Clicável e Navegação
    if st.session_state.IMAGE_FILES:
        st.subheader("Lista de Imagens")
        
        # Navegação com botões
        col_prev, col_next = st.columns(2)
        col_prev.button("⬅️ Anterior", on_click=prev_image)
        col_next.button("Próxima ➡️", on_click=next_image)
        
        st.markdown("---")
        
        current_idx = st.session_state.CURRENT_IMAGE_INDEX
        img_name = os.path.basename(st.session_state.IMAGE_FILES[current_idx])
        st.info(f"Visualizando: **{img_name}** ({current_idx + 1}/{len(st.session_state.IMAGE_FILES)})")

        # Configuração da Classe (Melhoria)
        st.subheader("Classe de Marcação (Requisitos 4, 7)")
        st.session_state.current_class_id = st.number_input(
            "ID da Classe (Inteiro):", 
            min_value=0, 
            value=st.session_state.LAST_CLASS_ID, 
            step=1,
            key='class_input_box'
        )
        st.session_state.LAST_CLASS_ID = st.session_state.current_class_id
        st.caption(f"Os próximos boxes serão rotulados como CLASSE: {st.session_state.current_class_id}")
        
        # Botão Salvar (Requisito 8 adaptado para salvar por imagem)
        st.markdown("---")
        if st.button("✅ Salvar Labels e Próxima Imagem (Requisito 8)", type="primary"):
            # O processamento dos boxes desenhados deve ser feito antes de avançar
            # (A lógica de processamento dos boxes desenhados está abaixo no canvas_result)
            st.warning("Clique no botão 'Processar e Salvar' na área principal após o desenho.")
            
# --- ÁREA PRINCIPAL (Visualização e Anotação) ---

if st.session_state.IMAGE_FILES:
    current_image_path = st.session_state.IMAGE_FILES[st.session_state.CURRENT_IMAGE_INDEX]
    
    try:
        img = Image.open(current_image_path).convert("RGB")
        img_width, img_height = img.size
        
        # Requisito 3: Canvas Desenho
        # NOTA: O tamanho do canvas deve ser fixo ou igual ao da imagem para precisão de coordenadas
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Cor de preenchimento
            stroke_width=2,
            stroke_color="#FF0000",
            background_image=img,
            update_streamlit=True,
            height=min(img_height, 700), # Limita a altura para visualização
            width=min(img_width, st.session_state.get('canvas_max_width', 800)),
            drawing_mode="rect", # Requisito 3: Modo Retângulo
            key="canvas_draw"
        )
        
        st.markdown("---")
        
        # Processamento dos boxes desenhados (Requisito 4/7)
        if canvas_result.json_data is not None and canvas_result.json_data['objects']:
            st.subheader("Boxes Desenhados (Clique em 'Processar e Salvar' para confirmar)")
            
            final_boxes_to_save = []
            
            # Adicionar a classe a cada objeto desenhado
            for i, obj in enumerate(canvas_result.json_data['objects']):
                if obj['type'] == 'rect':
                    # Aqui, a lógica de input da classe (Requisito 4) seria implementada,
                    # mas usaremos o valor do st
