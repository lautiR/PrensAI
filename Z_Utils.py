import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import chardet
import os

# ========================
# Función para obtener el texto plano de un link, manejando encoding
# ========================
def get_texto_plano_from_link(link):
    try:
        r = requests.get(link, timeout=10)
        if r.status_code == 200:
            # Intentar detectar el encoding si hay caracteres raros
            enc = r.encoding if r.encoding else 'utf-8'
            try:
                texto_bytes = r.content
                detected_enc = chardet.detect(texto_bytes)['encoding']
                enc = detected_enc if detected_enc else enc
                html = texto_bytes.decode(enc, errors='replace')
            except Exception:
                html = r.text  # Fallback
            soup = BeautifulSoup(html, 'html.parser')
            texto = soup.get_text(separator=' ', strip=True)
        else:
            texto = f"ERROR: Status code {r.status_code}"
    except Exception as e:
        texto = f"ERROR: {e}"
    return texto

# ========================
# Cargar Excel y procesar textos
# ========================
df = pd.read_excel("DataCollected/Super_Excel.xlsx")
df_sample = df  # Tomar solo las primeras 20 filas para test

textos = []
ids = []

# Crear carpeta de resultados si no existe
os.makedirs("Data_Results", exist_ok=True)

for idx, row in df_sample.iterrows():
    link = row['LINK']
    if pd.isna(link) or not str(link).startswith('http'):
        texto = 'ERROR: LINK VACÍO O INVÁLIDO'
    else:
        texto = get_texto_plano_from_link(link)
    textos.append(texto)
    ids.append(row['ID'])
    print(f"[{idx+1}/20] Procesado: {link}")
    time.sleep(0.5)  # Pequeño delay para no sobrecargar el servidor

# Agregar columna al dataframe original
df_sample = df_sample.copy()  # Esto elimina el warning
df_sample['TEXTO_PLANO'] = textos

# Guardar resultado en CSV y Excel

df_sample.to_csv("Data_Results/noticias_texto_20.csv", index=False, encoding='utf-8')
df_sample.to_excel("Data_Results/noticias_texto_20.xlsx", index=False)

print("¡Listo! Archivos guardados como noticias_texto_20.csv y noticias_texto_20.xlsx")
