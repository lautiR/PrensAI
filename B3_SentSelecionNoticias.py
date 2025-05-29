import pandas as pd
from transformers import pipeline, AutoTokenizer

# === Configuraciones ===
DATA_PATH = "Data_Results/dataset_IA.csv"
MAX_TOKENS = 120

# === Cargar dataset ===
df = pd.read_csv(DATA_PATH, encoding="utf-8")
df["VALORACION_NORM"] = df["VALORACIÓN"].fillna("").str.strip().str.upper()
df["CLASE_HUMANA"] = df["VALORACION_NORM"].map({"NEGATIVA": "NEGATIVA", "NEUTRA": "NEUTRA", "POSITIVA": "POSITIVA"})
df = df[df["CLASE_HUMANA"].notnull()].reset_index(drop=True)

# --- Lista de índices a analizar (EDITÁ ESTO) ---
indices_a_analizar = [1, 2, 3, 6, 9, 0]  # Cambiá por los índices que quieras

# === Cargar modelo ===
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="pysentimiento/robertuito-sentiment-analysis",
    tokenizer="pysentimiento/robertuito-sentiment-analysis",
    device=0, #0 gpu, -1 cpu
    batch_size=1
)
tokenizer = AutoTokenizer.from_pretrained("pysentimiento/robertuito-sentiment-analysis")

def sliding_window_debug(texto, pipeline, tokenizer, max_tokens=120):
    tokens = tokenizer.encode(texto, truncation=False)
    if len(tokens) < 10:
        return {
            'pred': 'OTRO',
            'neg_scores': [],
            'pos_scores': [],
            'chunks': 0,
            'skipped_short': 1,
            'skipped_long': 0,
            'chunk_info': []
        }
    neg_scores = []
    pos_scores = []
    chunk_info = []
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
            chunk_info.append({'text': decoded, 'label': result['label'], 'score': result['score']})
            if result['label'] == 'NEG':
                neg_scores.append(result['score'])
            if result['label'] == 'POS':
                pos_scores.append(result['score'])
        except Exception as e:
            chunk_info.append({'text': decoded, 'label': 'ERROR', 'score': 0.0, 'error': str(e)})
            continue
        total_chunks += 1
    if total_chunks == 0:
        return {
            'pred': 'OTRO',
            'neg_scores': neg_scores,
            'pos_scores': pos_scores,
            'chunks': total_chunks,
            'skipped_short': skipped_short,
            'skipped_long': skipped_long,
            'chunk_info': chunk_info
        }
    prom_neg = sum(neg_scores) / len(neg_scores) if neg_scores else 0.0
    prom_pos = sum(pos_scores) / len(pos_scores) if pos_scores else 0.0
    if len(neg_scores) > len(pos_scores):
        pred = 'NEG'
    elif len(neg_scores) == len(pos_scores):
        pred = 'NEG' if prom_neg > prom_pos else 'OTRO'
    else:
        pred = 'OTRO'
    return {
        'pred': pred,
        'neg_scores': neg_scores,
        'pos_scores': pos_scores,
        'chunks': total_chunks,
        'skipped_short': skipped_short,
        'skipped_long': skipped_long,
        'chunk_info': chunk_info,
        'prom_neg': prom_neg,
        'prom_pos': prom_pos
    }

# === Analizar y mostrar detalle para cada índice seleccionado ===
for idx in indices_a_analizar:
    if idx >= len(df):
        print(f"\n[!] Índice {idx} fuera de rango en el dataset.")
        continue
    row = df.iloc[idx]
    texto = str(row["TEXTO_PLANO"])
    clase_humana = row["CLASE_HUMANA"]
    resultado = sliding_window_debug(texto, sentiment_pipeline, tokenizer, MAX_TOKENS)
    print(f"\n{'='*60}\nNoticia idx={idx}")
    print(f"Fecha: {row['FECHA'] if 'FECHA' in df.columns else 'Sin FECHA'}")
    print(f"Valoración humana: {clase_humana}")
    print(f"Predicción modelo: {resultado['pred']}")
    print(f"Chunks válidos: {resultado['chunks']} | Cortos: {resultado['skipped_short']} | Largos: {resultado['skipped_long']}")
    print(f"Promedio scores NEG: {resultado.get('prom_neg', 0.0):.4f}, Promedio scores POS: {resultado.get('prom_pos', 0.0):.4f}")
    print(f"Texto (primeros 400 chars): {texto[:400]}\n")
    print("---- DETALLE DE CADA CHUNK ----")
    for i, info in enumerate(resultado['chunk_info']):
        resumen = info['text'][:100].replace('\n', ' ') + ("..." if len(info['text']) > 100 else "")
        if info['label'] == 'ERROR':
            print(f"[Chunk {i}] ERROR: {info['error']}")
        else:
            print(f"[Chunk {i}] Label: {info['label']} | Score: {info['score']:.4f} | Texto: {resumen}")
    print("="*60)
