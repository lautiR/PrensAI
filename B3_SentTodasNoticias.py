import pandas as pd
from transformers import pipeline, AutoTokenizer
from sklearn.metrics import confusion_matrix, classification_report
import time
import datetime

# === Configuraciones ===
DATA_PATH = "Data_Results/dataset_IA.csv"
MAX_TOKENS = 120

# === Cargar dataset ===
df = pd.read_csv(DATA_PATH, encoding="utf-8")
df["VALORACION_NORM"] = df["VALORACIÓN"].fillna("").str.strip().str.upper()
df["CLASE_HUMANA"] = df["VALORACION_NORM"].map({"NEGATIVA": "NEGATIVA", "NEUTRA": "NEUTRA", "POSITIVA": "POSITIVA"})
df = df[df["CLASE_HUMANA"].notnull()].reset_index(drop=True)  # Solo quedan bien clasificadas

print("Cantidad de ejemplos por clase humana:")
print(df["CLASE_HUMANA"].value_counts())

# === Cargar modelo ===
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="pysentimiento/robertuito-sentiment-analysis",
    tokenizer="pysentimiento/robertuito-sentiment-analysis",
    device=0, #0 usa gpu, -1 usa cpu
    batch_size=1
)
tokenizer = AutoTokenizer.from_pretrained("pysentimiento/robertuito-sentiment-analysis")

def sliding_window_stats(texto, pipeline, tokenizer, max_tokens=120):
    tokens = tokenizer.encode(texto, truncation=False)
    if len(tokens) < 10:
        return 'OTRO'
    neg_scores = []
    pos_scores = []
    total_chunks = 0
    for i in range(0, len(tokens), max_tokens):
        chunk = tokens[i:i+max_tokens]
        if len(chunk) < 10 or len(chunk) > max_tokens:
            continue
        decoded = tokenizer.decode(chunk, skip_special_tokens=True)
        re_tokenized = tokenizer.encode(decoded, truncation=False)
        if len(re_tokenized) > 128:
            continue
        try:
            result = pipeline(decoded)[0]
            if result['label'] == 'NEG':
                neg_scores.append(result['score'])
            if result['label'] == 'POS':
                pos_scores.append(result['score'])
        except Exception:
            continue
        total_chunks += 1
    if total_chunks == 0:
        return 'OTRO'
    if len(neg_scores) > len(pos_scores):
        return 'NEG'
    elif len(neg_scores) == len(pos_scores):
        return 'NEG' if (sum(neg_scores)/len(neg_scores) if neg_scores else 0) > (sum(pos_scores)/len(pos_scores) if pos_scores else 0) else 'OTRO'
    else:
        return 'OTRO'

def format_seconds(segundos):
    return str(datetime.timedelta(seconds=int(segundos)))

# === Analizar todo el dataset (solo con valoración válida) ===
labels_humano = df["CLASE_HUMANA"].tolist()
labels_modelo = []

start_time = time.time()
print("\n=== Progreso de inferencia ===")
for idx, row in df.iterrows():
    if (idx+1) % 50 == 0 or idx == 0 or (idx+1) == len(df):
        uptime = time.time() - start_time
        print(f"Procesando noticia {idx+1}/{len(df)}... (uptime: {format_seconds(uptime)})")
    pred = sliding_window_stats(str(row["TEXTO_PLANO"]), sentiment_pipeline, tokenizer, MAX_TOKENS)
    labels_modelo.append(pred)

df["PREDICTO"] = labels_modelo

# === Resumen general ===
print("\n=== DETALLE DE PREDICCIONES PARA CADA CLASE HUMANA ===")
for clase in ["NEGATIVA", "NEUTRA", "POSITIVA"]:
    total = (df["CLASE_HUMANA"] == clase).sum()
    como_neg = ((df["CLASE_HUMANA"] == clase) & (df["PREDICTO"] == "NEG")).sum()
    como_otro = ((df["CLASE_HUMANA"] == clase) & (df["PREDICTO"] == "OTRO")).sum()
    print(f"Clase {clase}: total={total} | predijo NEG={como_neg} ({100*como_neg/total:.2f}%), "
          f"OTRO={como_otro} ({100*como_otro/total:.2f}%)")

# También la versión más simple (NEG vs OTRO)
bin_labels_humano = ["NEG" if v=="NEGATIVA" else "OTRO" for v in labels_humano]
print("\n=== MATRIZ DE CONFUSIÓN BINARIA (NEG vs OTRO) ===")
print(confusion_matrix(bin_labels_humano, labels_modelo, labels=["NEG","OTRO"]))


uptime = time.time() - start_time
print(f"\nFIN DEL REPORTE (uptime total: {format_seconds(uptime)})")

# === Listado de índices de errores relevantes ===

# Falsos Negativos: humanas NEGATIVA, modelo OTRO
FN_NEGATIVAS_IDX = df.index[(df["CLASE_HUMANA"] == "NEGATIVA") & (df["PREDICTO"] == "OTRO")].tolist()
# Falsos Positivos: humanas NEUTRA, modelo NEG
FP_NEUTRAS_IDX = df.index[(df["CLASE_HUMANA"] == "NEUTRA") & (df["PREDICTO"] == "NEG")].tolist()
# Falsos Positivos: humanas POSITIVA, modelo NEG
FP_POSITIVAS_IDX = df.index[(df["CLASE_HUMANA"] == "POSITIVA") & (df["PREDICTO"] == "NEG")].tolist()

print("=== LISTADOS DE ÍNDICES POR ERROR DE CLASIFICACIÓN ===")
print("Falsos Negativos (índices de noticias NEGATIVAS humanas NO calificadas NEG por el modelo):")
print(FN_NEGATIVAS_IDX)
print("Falsos Positivos NEUTRAS (índices de noticias NEUTRAS humanas calificadas NEG por el modelo):")
print(FP_NEUTRAS_IDX)
print("Falsos Positivos POSITIVAS (índices de noticias POSITIVAS humanas calificadas NEG por el modelo):")
print(FP_POSITIVAS_IDX)

print("\n=== EJEMPLOS DE FALSOS NEGATIVOS ===")
for idx in FN_NEGATIVAS_IDX[:5]:
    print(f"\nIDX: {idx}")
    print(f"TEXTO (primeros 300 chars): {df.loc[idx, 'TEXTO_PLANO'][:300]}")
    print(f"VALORACION HUMANA: {df.loc[idx, 'CLASE_HUMANA']}")
    print(f"PREDICCIÓN MODELO: {df.loc[idx, 'PREDICTO']}")

print("\n=== EJEMPLOS DE FALSOS POSITIVOS EN POSITIVA ===")
for idx in FP_POSITIVAS_IDX[:5]:
    print(f"\nIDX: {idx}")
    print(f"TEXTO (primeros 300 chars): {df.loc[idx, 'TEXTO_PLANO'][:300]}")
    print(f"VALORACION HUMANA: {df.loc[idx, 'CLASE_HUMANA']}")
    print(f"PREDICCIÓN MODELO: {df.loc[idx, 'PREDICTO']}")

print("\n=== EJEMPLOS DE FALSOS POSITIVOS EN NEUTRA ===")
for idx in FP_NEUTRAS_IDX[:5]:
    print(f"\nIDX: {idx}")
    print(f"TEXTO (primeros 300 chars): {df.loc[idx, 'TEXTO_PLANO'][:300]}")
    print(f"VALORACION HUMANA: {df.loc[idx, 'CLASE_HUMANA']}")
    print(f"PREDICCIÓN MODELO: {df.loc[idx, 'PREDICTO']}")

# === Exportar a un solo Excel todos los errores relevantes ===
# Se concatena todo en orden: primero Falsos Negativos, luego FP en Positiva, luego FP en Neutra
columnas = ["FECHA", "TEXTO_PLANO", "CLASE_HUMANA", "PREDICTO"]

def build_df(indices, tipo_error):
    df_e = df.loc[indices, columnas].copy()
    df_e["INDICE_DATASET"] = indices
    df_e["TIPO_ERROR"] = tipo_error
    # Reordenar columnas: INDICE_DATASET primero
    cols_final = ["INDICE_DATASET"] + columnas + ["TIPO_ERROR"]
    return df_e[cols_final]

df_export = pd.concat([
    build_df(FN_NEGATIVAS_IDX, "FALSO_NEGATIVO_NEGATIVA"),
    build_df(FP_POSITIVAS_IDX, "FALSO_POSITIVO_POSITIVA"),
    build_df(FP_NEUTRAS_IDX, "FALSO_POSITIVO_NEUTRA")
], axis=0).reset_index(drop=True)
df_export.to_excel("errores_modelo_sentimiento.xlsx", index=False)
print("\nArchivo exportado: errores_modelo_sentimiento.xlsx")
