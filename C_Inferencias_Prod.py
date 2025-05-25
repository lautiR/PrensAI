# =====================================
# SCRIPT C_Inferencias: Inferencia por batch de links
# =====================================

import pandas as pd
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json

# -------- CONFIGURACIÓN --------
MODEL_PARENT_DIR = "Training_Results/"
# Elegí el modelo a usar según el score/carpeta que prefieras (poné el más reciente, o elegí a mano)
MODEL_FOLDER = sorted([f for f in os.listdir(MODEL_PARENT_DIR) if f.startswith("Modelo_")])[-1]
MODEL_PATH = os.path.join(MODEL_PARENT_DIR, MODEL_FOLDER)
INPUT_PATH = "Data_Results/links_a_analizar.xlsx"  # Archivo con columna LINK
OUTPUT_PATH = f"Resultados_Inferencias/resultados_inferencia_{MODEL_FOLDER}.xlsx"
os.makedirs("Resultados_Inferencias", exist_ok=True)

# ------- 1. CARGA DE MODELO Y TOKENIZER -------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
mlb_classes = np.load(os.path.join(MODEL_PATH, "mlb_classes.npy"), allow_pickle=True)
with open(os.path.join(MODEL_PATH, "metadata.json"), encoding="utf-8") as f:
    metadata = json.load(f)

# ------- 2. CARGA LINKS -------
df_links = pd.read_excel(INPUT_PATH)
if "LINK" not in df_links.columns:
    raise ValueError("El archivo debe tener columna LINK.")

# ------- 3. SCRAPING DE TEXTO -------
# VER: Si luego reificamos en utils.py (se repite en A_DATA)
from Z_Utils import get_texto_plano_from_link  # IMPORT DIRECTO DEL SCRIPT A

df_links["TEXTO_PLANO"] = df_links["LINK"].apply(get_texto_plano_from_link)

# ------- 4. INFERENCIA (por batch) -------
batch_size = 8
preds_all = []

model.eval()
with torch.no_grad():
    for i in range(0, len(df_links), batch_size):
        textos = df_links["TEXTO_PLANO"].iloc[i:i+batch_size].tolist()
        encodings = tokenizer(textos, truncation=True, padding=True, max_length=256, return_tensors="pt")
        outputs = model(**encodings)
        logits = outputs.logits.sigmoid().cpu().numpy()
        preds = (logits > 0.5).astype(int)
        preds_all.append(preds)

preds_final = np.vstack(preds_all)

# ------- 5. POSTPROCESADO Y EXPORT -------
# Convertir predicciones multilabel a columnas (ETIQUETA_X, MENCION_X, etc.)
for idx, label in enumerate(mlb_classes):
    df_links[label] = preds_final[:, idx]

# Opcional: convertir 1/0 en SI/NO para menciones
for label in mlb_classes:
    if label.upper().startswith("MENCION"):
        df_links[label] = df_links[label].map({1: "SI", 0: "NO"})

# Guardar Excel
print(f"Exportando resultados a {OUTPUT_PATH}")
df_links.to_excel(OUTPUT_PATH, index=False)
print(f"¡Listo! Resultados guardados en {OUTPUT_PATH}")
