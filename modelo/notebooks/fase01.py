import marimo

__generated_with = "0.17.7"
app = marimo.App(width="wide")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import io
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import zscore
    from sklearn.preprocessing import LabelEncoder, MinMaxScaler
    import mlflow
    import mlflow.keras
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
    from tensorflow.keras.optimizers import Adam
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import os

    plt.style.use("seaborn-v0_8-darkgrid")

    mo.md("#  Fase 1: Análisis, Preparación y Feature Engineering")
    return (
        mo,
        pd,
        np,
        io,
        plt,
        sns,
        zscore,
        LabelEncoder,
        MinMaxScaler,
    )



@app.cell
def _(mo, pd):
    df_raw = pd.read_csv(
        "C:/Users/samil/Desktop/APRENDIZAJE AUTOMATICO/PRIMER INTERCICLO/Practica-2-Aprendizaje-Automactico/data/dataset.csv"
    )

    mo.md("##  INFORMACIÓN GENERAL DEL DATASET")
    mo.md(f"""
    **Dimensiones del dataset:** {df_raw.shape[0]:,} filas x {df_raw.shape[1]} columnas  
    **Tamaño en memoria:** {df_raw.memory_usage(deep=True).sum() / 1024**2:.2f} MB
    """)
    mo.md("**Primeras 5 filas del dataset:**")
    return df_raw


@app.cell
def _(df_raw):
    df_raw.head()
    return


# =====================================================
# 3️⃣ INFORMACIÓN DE COLUMNAS Y NULOS
# =====================================================
@app.cell
def _(df_raw, mo, pd, io):
    mo.md("##  INFORMACIÓN DE COLUMNAS")

    _s = io.StringIO()
    df_raw.info(buf=_s)
    info_str = _s.getvalue()

    mo.md(f"```\n{info_str}\n```")

    null_summary = pd.DataFrame({
        "Valores Nulos": df_raw.isnull().sum(),
        "Porcentaje (%)": (df_raw.isnull().sum() / len(df_raw) * 100).round(2),
    })

    mo.md("### **Resumen de Valores Nulos**")
    if null_summary["Valores Nulos"].sum() == 0:
        mo.md("* No se encontraron valores nulos en el dataset.*")
    else:
        mo.md(null_summary.to_markdown())
    return df_raw


# =====================================================
# 4️⃣ CLASIFICACIÓN DE VARIABLES
# =====================================================
@app.cell
def _(df_raw, np, mo):
    numericas = df_raw.select_dtypes(include=[np.number]).columns.tolist()
    categoricas = df_raw.select_dtypes(include=["object"]).columns.tolist()

    mo.md(f"""
    ##  Clasificación de Variables
    **Numéricas ({len(numericas)}):** `{", ".join(numericas)}`
    
    **Categóricas ({len(categoricas)}):** `{", ".join(categoricas)}`
    """)
    return numericas, categoricas


# =====================================================
# 5️⃣ LIMPIEZA Y PREPARACIÓN DE DATOS
# =====================================================
@app.cell
def _(df_raw, pd, mo):
    mo.md("## Limpieza y Preparación de Datos")

    df_clean = df_raw.copy()

    # Eliminar columna ID
    if "id" in df_clean.columns:
        df_clean = df_clean.drop(columns=["id"])

    # Convertir columnas de fecha
    date_cols = [
        "created_at",
        "last_order_date",
        "last_stock_count_date",
        "expiration_date",
        "last_updated_at",
    ]
    for _col in date_cols:
        if _col in df_clean.columns:
            df_clean[_col] = pd.to_datetime(df_clean[_col], errors="coerce")

    # Manejar nulos (sin warnings)
    for _col in df_clean.columns:
        if df_clean[_col].dtype == "object":
            df_clean[_col] = df_clean[_col].fillna("Desconocido")
        else:
            df_clean[_col] = df_clean[_col].fillna(0)

    df_clean.drop_duplicates(inplace=True)
    df_clean.sort_values(by=["product_id", "created_at"], inplace=True)
    df_clean.reset_index(drop=True, inplace=True)

    mo.md(f" Datos limpios y ordenados: {df_clean.shape[0]} filas, {df_clean.shape[1]} columnas.")
    return df_clean



@app.cell
def _(df_clean, mo):
    mo.md("## Feature Engineering Temporal")

    df_feat = df_clean.copy()
    df_feat["semana_del_anio"] = df_feat["created_at"].dt.isocalendar().week
    df_feat["dia_del_mes"] = df_feat["created_at"].dt.day
    df_feat["dia_de_la_semana"] = df_feat["created_at"].dt.dayofweek
    df_feat["es_fin_de_semana"] = (df_feat["dia_de_la_semana"] >= 5).astype(int)
    df_feat["trimestre"] = df_feat["created_at"].dt.quarter

    mo.md("Se generaron las variables temporales correctamente.")
    return df_feat



@app.cell
def _(df_feat, np, mo):
    mo.md("##  Variables Históricas y de Tendencia")

    TARGET = "quantity_available"
    _group = df_feat.groupby("product_id")

    df_feat["variacion_stock_diaria"] = _group[TARGET].diff().fillna(0)
    df_feat["media_movil_7d"] = (
        _group[TARGET].rolling(window=7, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    df_feat["media_movil_30d"] = (
        _group[TARGET].rolling(window=30, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    df_feat["media_movil_exponencial"] = (
        _group[TARGET].ewm(span=7, adjust=False).mean().reset_index(level=0, drop=True)
    )
    df_feat["tendencia_stock"] = df_feat["media_movil_7d"] - df_feat["media_movil_30d"]
    df_feat["dias_desde_ultimo_pedido"] = (
        df_feat["created_at"] - df_feat["last_order_date"]
    ).dt.days
    df_feat["ratio_reservado_disponible"] = df_feat["quantity_reserved"] / (
        df_feat[TARGET] + 1e-6
    )

    mo.md("Variables de tendencia y ratios generadas correctamente.")
    return df_feat, TARGET


# =====================================================
# 8️⃣ LAGS Y ANOMALÍAS
# =====================================================
@app.cell
def _(df_feat, TARGET, zscore, np, mo):
    mo.md("## Lags y Detección de Anomalías")

    _group2 = df_feat.groupby("product_id")
    df_lags = df_feat.copy()

    df_lags["lag_1"] = _group2[TARGET].shift(1).bfill().fillna(0)
    df_lags["lag_7"] = _group2[TARGET].shift(7).bfill().fillna(0)
    df_lags["lag_30"] = _group2[TARGET].shift(30).bfill().fillna(0)

    df_lags["zscore"] = _group2[TARGET].transform(lambda x: zscore(x, nan_policy="omit"))
    df_lags["anomalia_stock"] = (np.abs(df_lags["zscore"]) > 3).astype(int)
    df_lags.drop(columns=["zscore"], inplace=True)
    df_lags.fillna(0, inplace=True)

    mo.md("Variables de lag y anomalías generadas exitosamente.")
    return df_lags



@app.cell
def _(df_lags, numericas, categoricas, TARGET, LabelEncoder, MinMaxScaler, mo):
    mo.md("## Codificación y Escalado de Variables")

    df_processed = df_lags.copy()

    # Copiamos listas para no sobrescribir las originales
    _numericas = [c for c in numericas if c in df_processed.columns and c != "id"]
    _categoricas = categoricas.copy()

    # Asegurar que 'product_id' sea categórica
    if "product_id" not in _categoricas:
        _categoricas.append("product_id")

    # Quitar el target de las numéricas
    if TARGET in _numericas:
        _numericas.remove(TARGET)

    # --- Label Encoding ---
    encoders = {}
    for _cat in _categoricas:
        if _cat in df_processed.columns:
            le = LabelEncoder()
            df_processed[_cat] = le.fit_transform(df_processed[_cat])
            encoders[_cat] = le

    # --- MinMax Scaling ---
    scaler_features = MinMaxScaler()
    scaler_target = MinMaxScaler()

    df_processed[_numericas] = scaler_features.fit_transform(df_processed[_numericas])
    df_processed[TARGET] = scaler_target.fit_transform(df_processed[[TARGET]])

    mo.md(f"""
    **Columnas escaladas:** {len(_numericas)}  
    **Columnas categóricas codificadas:** {len(_categoricas)}  
    **Shape final del dataset:** {df_processed.shape}
    """)

    return df_processed, _numericas, _categoricas




@app.cell
def _(df_raw, numericas, plt, mo):
    mo.md("##  Distribución de Variables de Inventario")

    inventory_vars = [
        "quantity_on_hand",
        "quantity_reserved",
        "quantity_available",
        "minimum_stock_level",
        "reorder_point",
        "optimal_stock_level",
    ]
    inventory_vars = [v for v in inventory_vars if v in numericas]

    if len(inventory_vars) == 0:
        mo.md("* No se encontraron variables de inventario.*")
    else:
        n_cols = 2
        n_rows = int(np.ceil(len(inventory_vars) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
        axes = axes.flatten()

        for idx, col in enumerate(inventory_vars):
            axes[idx].hist(df_raw[col], bins=40, color="#ff9999", edgecolor="black", alpha=0.8)
            axes[idx].axvline(df_raw[col].mean(), color="red", linestyle="--", linewidth=2)
            axes[idx].set_title(f"Distribución de {col}", fontsize=12, fontweight="bold")
            axes[idx].legend([f"Media: {df_raw[col].mean():.2f}"], loc="upper right")

        for i in range(len(inventory_vars), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        fig
    return



@app.cell
def _(mo):
    mo.md("""
    ---
     **Fase 01 completada correctamente**
    
    - No se detectaron valores nulos.  
    - Se generaron variables temporales, históricas y de tendencia.  
    - Se codificaron y escalaron las variables numéricas y categóricas.  
    - Distribuciones visualizadas correctamente.  
    
    Los datos están listos para la **Fase 02 (entrenamiento del modelo GRU/LSTM)**.
    """)
    return


if __name__ == "__main__":
    app.run()
