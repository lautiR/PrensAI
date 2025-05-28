import pandas as pd
from transformers import pipeline, AutoTokenizer
import matplotlib.pyplot as plt

# === Configuraciones ===
DATA_PATH = "Data_Results/dataset_IA.csv"
MAX_TOKENS = 120

# === Cargar dataset y modelo ===
df = pd.read_csv(DATA_PATH, encoding="utf-8")
df["VALORACION_NORM"] = df["VALORACIÓN"].fillna("").str.strip().str.upper()
df["NEGATIVA"] = df["VALORACION_NORM"] == "NEGATIVA"
df_sub = df.reset_index(drop=True)

# MOSTRAR LOS ÍNDICES DE NOTICIAS NEGATIVAS HUMANAS
indices_negativas = df_sub[df_sub["NEGATIVA"] == True].index.tolist()
print("Índices de noticias humanas NEGATIVAS:", indices_negativas)

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="pysentimiento/robertuito-sentiment-analysis",
    tokenizer="pysentimiento/robertuito-sentiment-analysis",
    device=-1,
    batch_size=1
)
tokenizer = AutoTokenizer.from_pretrained("pysentimiento/robertuito-sentiment-analysis")

# === Sliding window modificado ===
def sliding_window_stats(texto, pipeline, tokenizer, max_tokens=120):
    tokens = tokenizer.encode(texto, truncation=False)
    if len(tokens) < 10:
        return {'pred': 'OTRO', 'skipped_short': 1, 'skipped_long': 0, 'ponderacion_neg': 0.0,
                'prom_neg': 0.0, 'prom_pos': 0.0, 'diff': 0.0, 'chunks': 0}
    neg_scores = []
    pos_scores = []
    total_chunks = 0
    skipped_short = 0
    skipped_long = 0
    for i in range(0, len(tokens), max_tokens):
        chunk = tokens[i:i+max_tokens]
        if len(chunk) < 10:
            skipped_short += 1
            continue
        if len(chunk) > max_tokens:
            skipped_long += 1
            continue
        decoded = tokenizer.decode(chunk, skip_special_tokens=True)
        re_tokenized = tokenizer.encode(decoded, truncation=False)
        if len(re_tokenized) > 128:
            skipped_long += 1
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
        return {'pred': 'OTRO', 'skipped_short': skipped_short, 'skipped_long': skipped_long, 'ponderacion_neg': 0.0,
                'prom_neg': 0.0, 'prom_pos': 0.0, 'diff': 0.0, 'chunks': 0}
    prom_neg = sum(neg_scores)/len(neg_scores) if neg_scores else 0.0
    prom_pos = sum(pos_scores)/len(pos_scores) if pos_scores else 0.0
    diff = prom_neg - prom_pos
    if len(neg_scores) > len(pos_scores):
        pred = 'NEG'
    elif len(neg_scores) == len(pos_scores):
        pred = 'NEG' if prom_neg > prom_pos else 'OTRO'
    else:
        pred = 'OTRO'
    ponderacion_neg = sum(neg_scores) / total_chunks
    return {
        'pred': pred,
        'skipped_short': skipped_short,
        'skipped_long': skipped_long,
        'ponderacion_neg': ponderacion_neg,
        'prom_neg': prom_neg,
        'prom_pos': prom_pos,
        'diff': diff,
        'chunks': total_chunks
    }

# === Analizar todas las noticias negativas ===
res = []
for idx in indices_negativas:
    texto = df_sub.loc[idx, "TEXTO_PLANO"]
    print(f"\n--- Noticia idx={idx} ---")
    salida = sliding_window_stats(texto, sentiment_pipeline, tokenizer, MAX_TOKENS)
    salida['idx'] = idx
    res.append(salida)
    print(f"\nResultado PROMEDIO: {salida['pred']}")
    print(f"Chunks saltados: cortos={salida['skipped_short']}, largos={salida['skipped_long']}")
    print(f"Ponderación negativa (total chunks): {salida['ponderacion_neg']:.4f}")
    print(f"Promedio scores NEG: {salida['prom_neg']:.4f}, Promedio scores POS: {salida['prom_pos']:.4f}")
    print(f"Diff (NEG - POS): {salida['diff']:.4f}")

# === Métricas resumen ===
n_total = len(res)
n_neg = sum([1 for d in res if d['pred'] == 'NEG'])
p_neg = 100.0 * n_neg / n_total if n_total > 0 else 0.0

print(f"\nDe {n_total} noticias negativas humanas, el modelo clasificó {n_neg} como NEGATIVAS ({p_neg:.2f}%)\n")

# === Graficar y exportar ===
valores_diff = [d['diff'] for d in res]
plt.figure(figsize=(8,5))
plt.hist(valores_diff, bins=20, alpha=0.7)
plt.xlabel("Promedio NEG - Promedio POS")
plt.ylabel("Cantidad de noticias negativas (humanas)")
plt.title("Diferencia de promedios en noticias negativas (humanas)")
plt.tight_layout()
plt.savefig("diff_neg_pos_hist.png")
print("Histograma exportado como 'diff_neg_pos_hist.png'.")


# === ANALISIS ADICIONAL: 300 NOTICIAS NO NEGATIVAS ===
# Selecciona 300 noticias que NO son negativas humanas
indices_no_negativas = df_sub[df_sub["NEGATIVA"] == False].index[:300].tolist()

# Analiza esas 300 noticias extra
res_extra = []
for idx in indices_no_negativas:
    texto = df_sub.loc[idx, "TEXTO_PLANO"]
    salida = sliding_window_stats(texto, sentiment_pipeline, tokenizer, MAX_TOKENS)
    salida['idx'] = idx
    res_extra.append(salida)

# Junta ambos análisis
labels_humano = ['NEG' if idx in indices_negativas else 'OTRO' for idx in indices_negativas + indices_no_negativas]
labels_modelo = [d['pred'] for d in res + res_extra]

# Matriz de confusión
from sklearn.metrics import confusion_matrix, classification_report

print("\nMatriz de confusión (Modelo vs Humano): [NEG, OTRO]\n")
print(confusion_matrix(labels_humano, labels_modelo, labels=['NEG', 'OTRO']))
print("\nReporte de clasificación:\n")
print(classification_report(labels_humano, labels_modelo, digits=3))

# === Mostrar FP y FN ===
# FP: Modelo dice NEG, humano dice OTRO
# FN: Modelo dice OTRO, humano dice NEG
FP = []
FN = []
all_indices = indices_negativas + indices_no_negativas
for i, (h, m) in enumerate(zip(labels_humano, labels_modelo)):
    idx = all_indices[i]
    if h == 'OTRO' and m == 'NEG':
        FP.append(idx)
    if h == 'NEG' and m == 'OTRO':
        FN.append(idx)

print(f"\nFALSOS POSITIVOS (Modelo NEG, Humano OTRO): {len(FP)} casos.")
for fp_idx in FP[:10]:  # Mostramos hasta 10 para no saturar
    texto_fp = str(df_sub.loc[fp_idx, 'TEXTO_PLANO']).replace('\n',' ')
    print(f"FP idx={fp_idx}, texto: {texto_fp[:150]}...")

print(f"\nFALSOS NEGATIVOS (Modelo OTRO, Humano NEG): {len(FN)} casos.")
for fn_idx in FN[:10]:
    texto_fn = str(df_sub.loc[fn_idx, 'TEXTO_PLANO']).replace('\n',' ')
    print(f"FN idx={fn_idx}, texto: {texto_fn[:150]}...")

print("\n--- FIN DIAGNÓSTICO FP/FN ---")
# === Resumen final de matriz de confusión ===
mat = confusion_matrix(labels_humano, labels_modelo, labels=['NEG', 'OTRO'])
TP = mat[0,0]  # humanas NEG, modelo NEG
FN = mat[0,1]  # humanas NEG, modelo OTRO
FP = mat[1,0]  # humanas OTRO, modelo NEG
TN = mat[1,1]  # humanas OTRO, modelo OTRO
total = TP + TN + FP + FN

print("\n========== RESUMEN FINAL DE LA CORRIDA ==========")
print(f"Total de casos evaluados: {total}")
print(f"TP (humanas NEG, modelo NEG): {TP}")
print(f"FN (humanas NEG, modelo OTRO): {FN}")
print(f"FP (humanas OTRO, modelo NEG): {FP}")
print(f"TN (humanas OTRO, modelo OTRO): {TN}")
print(f"Accuracy global: {100.0 * (TP + TN) / total:.2f}%")
print(f"Sensibilidad (Recall NEG): {100.0 * TP / (TP + FN):.2f}%" if (TP+FN)>0 else "Sensibilidad: N/A")
print(f"Especificidad (Recall OTRO): {100.0 * TN / (TN + FP):.2f}%" if (TN+FP)>0 else "Especificidad: N/A")
print("==============================================\n")
