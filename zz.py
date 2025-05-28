import pandas as pd
from transformers import pipeline, AutoTokenizer
import matplotlib.pyplot as plt

# === Configuraciones ===
DATA_PATH = "Data_Results/dataset_IA.csv"
MAX_TOKENS = 120
UMBRAL_PROMEDIO = 0.4

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
def sliding_window_stats(texto, pipeline, tokenizer, max_tokens=120, min_ponderacion=0.5):
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
    # Promedios:
    prom_neg = sum(neg_scores)/len(neg_scores) if neg_scores else 0.0
    prom_pos = sum(pos_scores)/len(pos_scores) if pos_scores else 0.0
    diff = prom_neg - prom_pos
    # Lógica nueva:
    if len(neg_scores) > len(pos_scores):
        pred = 'NEG'
    elif len(neg_scores) == len(pos_scores):
        pred = 'NEG' if prom_neg > prom_pos else 'OTRO'
    else:
        pred = 'OTRO'
    # Ponderación negativa sobre total de chunks (no afecta más la predicción)
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
    salida = sliding_window_stats(texto, sentiment_pipeline, tokenizer, MAX_TOKENS, UMBRAL_PROMEDIO)
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

print(f"\nPromedio de (NEG - POS) en negativas humanas: {sum(valores_diff)/len(valores_diff):.4f}")
print(f"Máximo: {max(valores_diff):.4f}, Mínimo: {min(valores_diff):.4f}")
