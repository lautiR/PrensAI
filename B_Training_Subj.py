# =====================================
# SCRIPT B_Training_BinSent: Fine-tune modelo binario de sentimiento (NEGATIVO vs OTRO)
# =====================================
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import os
import json
from datetime import datetime
import transformers

print("Transformers version:", transformers.__version__)
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    print(f"¡Entrenando en GPU! ({device_name})")
else:
    print("Entrenando en CPU (NO se detectó GPU)")

# -------- CONFIG ---------
DATA_PATH = "Data_Results/dataset_IA.csv"   # Ajusta si lo necesitás
MODEL_NAME = "PlanTL-GOB-ES/roberta-base-bne"
TRAINING_RESULTS_DIR = "Training_Results/"
os.makedirs(TRAINING_RESULTS_DIR, exist_ok=True)
INPUT_COL = "TEXTO_PLANO"
TARGET_COL = "VALORACIÓN"    # Usar SOLO esta columna

# --------- 1. CARGA Y PREPRO ---------
df = pd.read_csv(DATA_PATH, encoding="utf-8")
df = df.dropna(subset=[INPUT_COL, TARGET_COL])

# --------- 2. BINARIZACIÓN DE TARGET ---------
def binarizar_valoracion(val):
    val = str(val).strip().upper()
    return 1 if val == "NEGATIVO" else 0

df["target"] = df[TARGET_COL].apply(binarizar_valoracion)

print("Distribución de clases (0 = OTRO, 1 = NEGATIVO):")
print(df["target"].value_counts())

# --------- 3. SPLIT TEMPORAL ---------
df["FECHA"] = pd.to_datetime(df["FECHA"], errors="coerce")
df = df.sort_values("FECHA")
split_index = int(len(df) * 0.85)
X_train = df[INPUT_COL].iloc[:split_index].tolist()
Y_train = df["target"].iloc[:split_index].tolist()
X_test = df[INPUT_COL].iloc[split_index:].tolist()
Y_test = df["target"].iloc[split_index:].tolist()

# --------- 4. TOKENIZACIÓN ---------
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

# --------- 5. MODELO ---------
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=1,  # Binario
    problem_type="single_label_classification"
)

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
metrics_dict = {}

def compute_metrics(pred):
    preds = pred.predictions.squeeze() > 0    # Logits > 0 → 1
    labels = pred.label_ids
    f1 = f1_score(labels, preds)
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds)
    rec = recall_score(labels, preds)
    metrics_dict.update({'f1': f1, 'accuracy': acc, 'precision': prec, 'recall': rec})
    return {"f1": f1, "accuracy": acc, "precision": prec, "recall": rec}

training_args = TrainingArguments(
    output_dir=TRAINING_RESULTS_DIR,
    num_train_epochs=20,  # Early stopping
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir=TRAINING_RESULTS_DIR + "logs/",
    logging_steps=50,
    do_train=True,
    do_eval=True,
    evaluation_strategy="epoch",
    save_strategy="no"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# --------- 6. ENTRENAMIENTO CON EARLY STOPPING MANUAL ---------
print("\n======= INICIO DE ENTRENAMIENTO BINARIO =======\n")
best_f1 = 0
patience = 7
epochs_no_improve = 0
best_state_dict = None

for epoch in range(training_args.num_train_epochs):
    trainer.train()
    eval_results = trainer.evaluate()
    current_f1 = eval_results["eval_f1"]
    print(f"[Epoch {epoch+1}] F1: {current_f1:.4f} | Best: {best_f1:.4f}")
    if current_f1 > best_f1:
        best_f1 = current_f1
        epochs_no_improve = 0
        best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}  # CPU por si GPU
        print(f"\U0001F3C5 ¡Nuevo mejor modelo (F1 {best_f1:.4f}) guardado!")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"⏹️ Early stopping: no mejora en {patience} épocas.")
            break

if best_state_dict:
    model.load_state_dict(best_state_dict)

# --------- 7. GUARDADO FINAL ---------
f1_score_final = best_f1
acc_final = metrics_dict.get('accuracy', 0)
time_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_folder = f"ModeloBin_F1_{f1_score_final:.3f}_ACC_{acc_final:.3f}_{time_stamp}"
full_model_path = os.path.join(TRAINING_RESULTS_DIR, model_folder)
os.makedirs(full_model_path, exist_ok=True)
trainer.save_model(full_model_path)
tokenizer.save_pretrained(full_model_path)

# Guardar metadata
metadata = {
    "datetime": time_stamp,
    "f1_score": float(f1_score_final),
    "accuracy": float(acc_final),
    "model_name": MODEL_NAME,
    "input_col": INPUT_COL,
    "target_col": TARGET_COL,
    "params": training_args.to_dict(),
    "patience": patience,
}
with open(os.path.join(full_model_path, "metadata.json"), "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=4)

print("\n¡Entrenamiento finalizado! Modelo y tokenizer guardados en", full_model_path)
print("\n**Input:**", INPUT_COL)
print("**Target binario:** NEGATIVO (1) vs OTRO (0)")
print("**Mejor F1:**", f1_score_final)
print("**Accuracy final:**", acc_final)
