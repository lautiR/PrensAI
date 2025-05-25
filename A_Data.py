# ===============================================
# SCRIPT A_DATA: Prepara dataset limpio para IA
# ===============================================
import pandas as pd
import os

# Ruta al CSV con texto plano generado (todas las filas)
DATA_PATH = 'Data_Results/noticias_texto_30k_limpio.csv'   # Cambia el nombre si el tuyo es diferente
EXPORT_PATH = 'Data_Results/dataset_IA.csv'
EXPORT_PATH_XLSX = 'Data_Results/dataset_IA.xlsx'

# ---------------------------
# CAMPOS OBLIGATORIOS/FIJOS
# ---------------------------
CAMPOS_FIJOS = [
    'TITULO', 'TIPO DE PUBLICACION', 'FECHA', 'SOPORTE', 'MEDIO', 'PROGRAMA / SECCIÓN',
    'CONDUCTOR / AUTOR', 'ENTREVISTADO', 'EVENTO / TEMA', 'LINK',
    'ALCANCE/AUDIENCIA/IMPACTO', 'COTIZACION', 'TAPA', 'VALORACIÓN', 'EJE COMUNICACIONAL',
    'TEXTO_PLANO'   # Campo fundamental para la IA
]

# ----------------------------
# Cargar el dataset completo
# ----------------------------
df = pd.read_csv(DATA_PATH, encoding='utf-8')

# ----------------------------
# DETECTAR columnas de MENCIONES y ETIQUETAS
# ----------------------------
col_menciones = [c for c in df.columns if 'Mencion' in c or 'Mención' in c]
col_etiquetas = [c for c in df.columns if 'ETIQUETA' in c.upper()]

# Normalizar menciones (SI/NO, string limpio)
def normalizar_si_no(val):
    val = str(val).strip().upper()
    return 'SI' if val == 'SI' else 'NO'

df_menciones = df[col_menciones].applymap(normalizar_si_no)

# Normalizar etiquetas, dejando sólo 3 etiquetas por noticia, resto se ignora
MAX_ETIQUETAS = 3
df_etiquetas = df[col_etiquetas].fillna("").astype(str)

# (Opcional) Renombrar las columnas de etiquetas para que sean ETIQUETA_1, 2, 3
for i in range(MAX_ETIQUETAS):
    if i < len(df_etiquetas.columns):
        df_etiquetas = df_etiquetas.rename(columns={df_etiquetas.columns[i]: f'ETIQUETA_{i+1}'})
    else:
        df_etiquetas[f'ETIQUETA_{i+1}'] = ""

# ----------------------------
# UNIR TODO EN UN SOLO DATAFRAME
# ----------------------------
df_out = df[CAMPOS_FIJOS].copy()
for col in df_menciones.columns:
    df_out[col] = df_menciones[col]
for col in [f'ETIQUETA_{i+1}' for i in range(MAX_ETIQUETAS)]:
    df_out[col] = df_etiquetas[col] if col in df_etiquetas.columns else ""

# (Opcional) Reordenar columnas a gusto
df_out = df_out[ CAMPOS_FIJOS + list(df_menciones.columns) + [f'ETIQUETA_{i+1}' for i in range(MAX_ETIQUETAS)] ]

# ----------------------------
# GUARDAR DATASET LIMPIO FINAL
# ----------------------------
os.makedirs('Data_Results', exist_ok=True)
df_out.to_csv(EXPORT_PATH, index=False, encoding='utf-8')
df_out.to_excel(EXPORT_PATH_XLSX, index=False)

print(f"¡Listo! Dataset de IA guardado en {EXPORT_PATH} y {EXPORT_PATH_XLSX}")

# --------- Notas ----------
# - Si necesitás eliminar/ignorar algún campo, editá CAMPOS_FIJOS arriba.
# - Los campos FIJOS son aquellos que no se pueden aprender fácil por IA porque dependen de reglas raras o lógica de negocio,
#   pero en general, en tu caso la mayoría pueden ir como targets de la IA. Sólo los que tengan reglas tipo CRISIS, GESTIONADA SIN MENCIÓN REAL, ETC,
#   se excluyen (eso lo documentás y los quitás manual si querés).
