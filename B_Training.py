# =====================================
# SCRIPT B_Training: Fine-tune modelo IA
# =====================================

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import os
import json
from datetime import datetime

# ------- Chequeo de GPU -------
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    print(f"¡Entrenando en GPU! ({device_name})")
else:
    print("Entrenando en CPU (NO se detectó GPU)")

# -------- CONFIG ---------
DATA_PATH = "Data_Results/dataset_IA.csv"
MODEL_NAME = "dccuchile/bert-base-spanish-wwm-uncased"  # BETO (puede cambiarse por otro de HF)
TRAINING_RESULTS_DIR = "Training_Results/"
os.makedirs(TRAINING_RESULTS_DIR, exist_ok=True)

TARGETS_ETIQUETAS = ["ETIQUETA_1", "ETIQUETA_2", "ETIQUETA_3"]  # Se pueden sumar menciones, valoración, etc.
INPUT_COL = "TEXTO_PLANO"

# --------- 1. CARGA Y PREPRO ---------
df = pd.read_csv(DATA_PATH, encoding="utf-8")
df = df.dropna(subset=[INPUT_COL])

# --- CHEQUEO Y BORRADO DE COLUMNAS ETIQUETA VACÍAS ---
for col in TARGETS_ETIQUETAS:
    if df[col].fillna("").str.strip().eq("").all():
        print(f"Advertencia: columna {col} está completamente vacía y será removida.")
        df = df.drop(columns=[col])
# Actualiza lista real de targets
TARGETS_ETIQUETAS = [c for c in TARGETS_ETIQUETAS if c in df.columns]

# --- SPLIT TEMPORAL: Test = noticias más recientes ---
# Convertir FECHA a datetime para ordenar correctamente (modifica el nombre si tu columna se llama distinto)
df["FECHA"] = pd.to_datetime(df["FECHA"], errors="coerce")
df = df.sort_values("FECHA")

def fila_a_lista(row):
    return [x for x in row[TARGETS_ETIQUETAS] if isinstance(x, str) and x.strip()]

y = df.apply(fila_a_lista, axis=1).tolist()
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(y)  # Matriz N x etiquetas_distintas

print("Clases multilabel (targets) que va a aprender el modelo:")
print(list(mlb.classes_))
print("Shape Y (N, num_labels):", Y.shape)
print("Cantidad de ejemplos positivos por clase (en orden):")
print(Y.sum(axis=0))

split_index = int(len(df) * 0.85)
X_train = df[INPUT_COL].iloc[:split_index].tolist()
Y_train = Y[:split_index]
X_test = df[INPUT_COL].iloc[split_index:].tolist()
Y_test = Y[split_index:]


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=256)
test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=256)

class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx]).float()
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = NewsDataset(train_encodings, Y_train)
test_dataset = NewsDataset(test_encodings, Y_test)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=Y.shape[1],
    problem_type="multi_label_classification"
)

from sklearn.metrics import f1_score, accuracy_score
metrics_dict = {}

def compute_metrics(pred):
    preds = pred.predictions > 0.5
    labels = pred.label_ids
    f1 = f1_score(labels, preds, average="micro")
    acc = accuracy_score(labels, preds)
    # Guardar para resumen
    metrics_dict['f1'] = f1
    metrics_dict['accuracy'] = acc
    return {"f1": f1, "accuracy": acc}

training_args = TrainingArguments(
    output_dir=TRAINING_RESULTS_DIR,
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir=TRAINING_RESULTS_DIR + "logs/",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

# --------- 9. GUARDAR MODELO EN CARPETA ÚNICA ---------

# Carpeta por score: Modelo_ScoreXX_TIMESTAMP
f1_score_final = metrics_dict.get('f1', 0)
acc_final = metrics_dict.get('accuracy', 0)
time_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_folder = f"Modelo_F1_{f1_score_final:.3f}_ACC_{acc_final:.3f}_{time_stamp}"
full_model_path = os.path.join(TRAINING_RESULTS_DIR, model_folder)
os.makedirs(full_model_path, exist_ok=True)

# Guardar modelo y tokenizer
trainer.save_model(full_model_path)
tokenizer.save_pretrained(full_model_path)
np.save(os.path.join(full_model_path, "mlb_classes.npy"), mlb.classes_)

# Guardar metadata
metadata = {
    "datetime": time_stamp,
    "f1_score": float(f1_score_final),
    "accuracy": float(acc_final),
    "model_name": MODEL_NAME,
    "labels": mlb.classes_.tolist(),
    "input_col": INPUT_COL,
    "targets": TARGETS_ETIQUETAS,
    "params": training_args.to_dict(),
}
with open(os.path.join(full_model_path, "metadata.json"), "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=4)

# Guardar histórico
csv_hist = os.path.join(TRAINING_RESULTS_DIR, "training_history.csv")
hist_line = {
    "datetime": time_stamp,
    "model_folder": model_folder,
    "f1_score": f1_score_final,
    "accuracy": acc_final,
    "model_name": MODEL_NAME,
    "n_train": len(X_train),
    "n_test": len(X_test)
}
hist_exists = os.path.isfile(csv_hist)
df_hist = pd.DataFrame([hist_line])
df_hist.to_csv(csv_hist, mode='a', index=False, header=not hist_exists)

print(f"¡Entrenamiento finalizado! Modelo, tokenizer y archivos guardados en {full_model_path}")
print(f"Histórico actualizado en {csv_hist}")
