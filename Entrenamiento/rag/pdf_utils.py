# pdf_utils.py
"""
Utilidades para procesar PDFs y cargarlos en la base de datos vectorial
"""
import os
import PyPDF2
import re
from tqdm import tqdm

def extract_text_from_pdf(pdf_path):
    """
    Extrae el texto completo de un archivo PDF
    
    Args:
        pdf_path: Ruta al archivo PDF
        
    Returns:
        Texto extraído del PDF
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"El archivo {pdf_path} no existe")
        
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        num_pages = len(reader.pages)
        
        print(f"Procesando PDF con {num_pages} páginas...")
        for page_num in tqdm(range(num_pages)):
            page = reader.pages[page_num]
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
                    
    return text

def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    """
    Divide el texto en fragmentos más pequeños con superposición
    
    Args:
        text: Texto completo a dividir
        chunk_size: Tamaño de cada fragmento
        chunk_overlap: Superposición entre fragmentos
        
    Returns:
        Lista de fragmentos de texto
    """
    chunks = []
    
    # Eliminar espacios en blanco excesivos
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Dividir el texto en párrafos
    paragraphs = text.split('\n\n')
    
    current_chunk = ""
    for paragraph in paragraphs:
        # Si añadir el párrafo excede el tamaño del fragmento,
        # guarda el fragmento actual y comienza uno nuevo
        if len(current_chunk) + len(paragraph) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # Si el párrafo es más grande que el tamaño del fragmento,
            # dividirlo en fragmentos más pequeños
            if len(paragraph) > chunk_size:
                words = paragraph.split()
                current_chunk = ""
                for word in words:
                    if len(current_chunk) + len(word) > chunk_size:
                        chunks.append(current_chunk.strip())
                        # Mantener algo de contexto con la superposición
                        overlap_words = current_chunk.split()[-int(chunk_overlap/10):]
                        current_chunk = " ".join(overlap_words) + " "
                    current_chunk += word + " "
            else:
                current_chunk = paragraph + " "
        else:
            current_chunk += paragraph + " "
    
    # Añadir el último fragmento si no está vacío
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
        
    print(f"Texto dividido en {len(chunks)} fragmentos")
    return chunks

def load_pdf_to_db(pdf_path, model_manager, vector_db, chunk_size=1000, chunk_overlap=200):
    """
    Carga un PDF en la base de datos vectorial
    
    Args:
        pdf_path: Ruta al archivo PDF
        model_manager: Instancia de ModelManager para generar embeddings
        vector_db: Instancia de VectorDatabase para almacenar documentos
        chunk_size: Tamaño de cada fragmento
        chunk_overlap: Superposición entre fragmentos
        
    Returns:
        Número de fragmentos añadidos
    """
    # Extraer texto del PDF
    text = extract_text_from_pdf(pdf_path)
    
    # Dividir en fragmentos
    chunks = chunk_text(text, chunk_size, chunk_overlap)
    
    # Obtener metadatos del PDF
    pdf_filename = os.path.basename(pdf_path)
    
    # Añadir fragmentos a la base de datos
    print(f"Añadiendo {len(chunks)} fragmentos a la base de datos vectorial...")
    for i, chunk in enumerate(chunks):
        metadata = {
            "source": pdf_filename,
            "chunk_id": i,
            "total_chunks": len(chunks),
            "type": "pdf"
        }
        
        # Generar embedding y añadir a la BD
        embedding = model_manager.generate_embeddings(chunk)
        vector_db.add_document(chunk, embedding, metadata)
    
    # Guardar la BD
    vector_db.save()
    
    print(f"PDF procesado: {len(chunks)} fragmentos añadidos a la base de datos")
    return len(chunks)