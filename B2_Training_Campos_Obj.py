# =========================================
# SCRIPT C_NER_Preparacion: Generar dataset NER para Transformers
# =========================================
import pandas as pd
import os
import json

# ---- CONFIG ----
DATA_PATH = 'Data_Results/dataset_IA.csv'
OUTPUT_PATH = 'Data_Results/dataset_NER.jsonl'

# ---- CAMPOS DE ENTIDAD (todos menos subjetivos y texto plano) ----
TARGETS_SUBJETIVOS = [
    "EVENTO / TEMA", "ETIQUETA", "ETIQUETA 2", "VALORACIÓN", "ÁREA", "MediaApoyo"
]
INPUT_COL = "TEXTO_PLANO"

# Leemos el CSV
print(f"Leyendo dataset: {DATA_PATH}")
df = pd.read_csv(DATA_PATH, encoding='utf-8')

# Definir qué campos queremos buscar como entidades (todo menos subjetivos y TEXTO_PLANO)
CAMPOS_ENTIDAD = [
    c for c in df.columns
    if c not in [INPUT_COL] + TARGETS_SUBJETIVOS and not c.startswith("ID")
]

# Mapeo especial para columnas tipo "Mención Jorge Macri" => nombre para buscar
MAP_MENCION = {
    c: c.replace("Mención ", "").replace("Mencion ", "")
    for c in df.columns if "Mención" in c or "Mencion" in c
}

# Para debug, cuántas entidades encuentra vs. esperadas
stats_entidades = {k: 0 for k in CAMPOS_ENTIDAD}

# Función para buscar span (start, end) de un valor dentro del texto
# Lo hace "fuzzy", tolerando tildes y espacios extra
import unicodedata

def normalize_text(t):
    if pd.isna(t):
        return ""
    t = str(t)
    t = unicodedata.normalize('NFKD', t)
    t = ''.join([c for c in t if not unicodedata.combining(c)])
    t = t.lower().strip()
    return t

def find_span(text, value):
    # Busca el valor dentro del texto, versión normalizada
    n_text = normalize_text(text)
    n_value = normalize_text(value)
    if not n_value or n_value == "nan":
        return None
    start = n_text.find(n_value)
    if start == -1:
        return None
    # Ahora mapeamos al texto original (puede ser distinto por tildes)
    # Tolerancia: buscamos el substring en texto original cerca de la posición encontrada
    approx = value.strip()
    idx = text.lower().find(approx.lower())
    if idx != -1:
        return idx, idx + len(approx)
    else:
        # Último recurso, devolvemos el índice aproximado de la normalizada
        return None

data = []
for i, row in df.iterrows():
    text = str(row[INPUT_COL])
    entities = []
    for c in CAMPOS_ENTIDAD:
        val = str(row[c]) if c in row else ""
        if not val or val == "nan" or val.strip() == "":
            continue
        # Menciones: solo si valor es SI, buscamos el nombre
        if c in MAP_MENCION:
            if val.upper() == "SI":
                nombre = MAP_MENCION[c]
                span = find_span(text, nombre)
                if span:
                    entities.append([span[0], span[1], c])
                    stats_entidades[c] += 1
            continue
        # Para los demás, buscamos el valor exacto
        span = find_span(text, val)
        if span:
            entities.append([span[0], span[1], c])
            stats_entidades[c] += 1
        else:
            # Si no lo encuentra, se ignora (se puede loguear)
            pass
    if entities:
        data.append({"text": text, "entities": entities})

# Guardar como JSONL para NER
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for item in data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"¡Listo! Dataset NER guardado en {OUTPUT_PATH}")
print(f"Cantidad de registros con entidades: {len(data)}")
print("Distribución de entidades por campo:")
for k, v in stats_entidades.items():
    print(f"  {k}: {v}")
