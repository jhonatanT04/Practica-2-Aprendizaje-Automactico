import marimo

__generated_with = "0.17.7"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # Fase 2: Modelado y Evaluación con MLflow

    En esta fase, usaremos el dataset procesado (`dataset_processed_advanced.csv`)
    para entrenar un modelo de Machine Learning.

    El objetivo es predecir `quantity_available`.

    Usaremos **MLflow** para registrar y gestionar nuestros experimentos,
    tal como lo solicitó el PM.
    """)
    return


@app.cell
def _():

    import pandas as pd
    import numpy as np


    import mlflow
    import mlflow.sklearn 


    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


    import matplotlib.pyplot as plt
    import seaborn as sns
    import joblib 


    sns.set_theme(style="whitegrid")
    pd.set_option('display.float_format', lambda x: '%.3f' % x)

    print("Todas las librerías para modelado y MLflow han sido importadas.")
    return (
        RandomForestRegressor,
        joblib,
        mean_absolute_error,
        mean_squared_error,
        mlflow,
        np,
        pd,
        plt,
        r2_score,
        sns,
        train_test_split,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Carga de Datos Procesados

    Cargamos el dataset `dataset_processed_advanced.csv` que creamos en la Fase 2.
    Este dataset ya está 100% limpio, es numérico y está listo para el modelo.
    """)
    return


@app.cell
def _(pd):
    # Cargar el dataset final de la Fase 1
    try:
        df_model = pd.read_csv("C:/Users/samil/Desktop/APRENDIZAJE AUTOMATICO/PRIMER INTERCICLO/Practica-2-Aprendizaje-Automactico/data/dataset_processed_advanced.csv")

        print(f"Dataset cargado exitosamente. Forma: {df_model.shape}")
        print("\n--- df_model.info() ---")
        df_model.info()

    except FileNotFoundError:
        print("Error: No se encontró el archivo 'dataset_processed_advanced.csv'.")
        print("Asegúrate de que el archivo exista en la ruta especificada.")
        df_model = pd.DataFrame() # Crear un df vacío para evitar errores

    return (df_model,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Definición de Variables (X, y)

    Separamos nuestro dataset en:
    * **`y` (Target):** La variable que queremos predecir (`quantity_available`).
    * **`X` (Features):** Todas las "pistas" (lags, medias, etc.) que usará el modelo.
    """)
    return


@app.cell
def _(df_model):
    if not df_model.empty:
        # 1. Definir el Target (la respuesta)
        y = df_model['quantity_available']

        # 2. Definir los Features (las pistas)
        #    Además de 'quantity_available' y 'product_sku',
        #    también eliminamos 'region_almacen' (que contiene "Sur").
        
        columnas_a_excluir = ['quantity_available', 'product_sku']
        
        # Añadir 'region_almacen' a la lista de exclusión SI existe
        if 'region_almacen' in df_model.columns:
            columnas_a_excluir.append('region_almacen')
        
        X = df_model.drop(columns=columnas_a_excluir)

        print(f"Target 'y' definido. (Total: {y.shape[0]} registros)")
        print(f"Features 'X' definidos. (Total: {X.shape[0]} registros, {X.shape[1]} features)")
    else:
        print("Dataset vacío, no se pueden definir X e y.")
        X, y = None, None
    return X, y


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. División de Datos (Train/Test Split)

    Dividimos los datos en dos conjuntos:
    * **Entrenamiento (80%):** Los datos que usamos para "enseñarle" al modelo.
    * **Prueba (20%):** Los datos que "escondemos" para evaluar qué tan bien
        aprendió el modelo con datos que nunca ha visto.
    """)
    return


@app.cell
def _(X, train_test_split, y):
    if X is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2,    # 20% de los datos para prueba
            random_state=42   # 'random_state' asegura que la división sea siempre la misma
        )

        print(f"Datos de Entrenamiento (Train): {X_train.shape[0]} registros")
        print(f"Datos de Prueba (Test): {X_test.shape[0]} registros")
    else:
        print("No se pueden dividir los datos porque el dataset está vacío.")
        X_train, X_test, y_train, y_test = [None] * 4

    return X_test, X_train, y_test, y_train


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Configuración del Experimento MLflow

    Aquí le decimos a MLflow dónde guardar los resultados.
    * `set_tracking_uri`: Crea una carpeta local `mlruns` para guardar todo.
    * `set_experiment`: Nombra nuestro "cuaderno de laboratorio".
    """)
    return


@app.cell
def _(mlflow):
    # 1. Definir dónde se guardarán los logs (creará una carpeta 'mlruns')
    mlflow.set_tracking_uri("mlruns")

    # 2. Ponerle un nombre a nuestro conjunto de experimentos
    experiment_name = "Prediccion de Stock"
    mlflow.set_experiment(experiment_name)

    print(f"MLflow configurado. Los experimentos se guardarán en la carpeta 'mlruns'.")
    print(f"Nombre del experimento: '{experiment_name}'")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Entrenamiento y Registro con MLflow

    1.  `with mlflow.start_run()`: Inicia una nueva "sesión" de registro.
    2.  Definimos el modelo (RandomForest).
    3.  `mlflow.log_params()`: Registra la configuración del modelo.
    4.  `model.fit()`: Entrenamos el modelo.
    5.  `model.predict()`: Hacemos predicciones.
    6.  `mlflow.log_metrics()`: Registra los resultados (MAE, MSE, R²).
    7.  `mlflow.sklearn.log_model()`: Guarda el archivo del modelo.
    """)
    return


@app.cell
def _(
    RandomForestRegressor,
    X_test,
    X_train,
    mean_absolute_error,
    mean_squared_error,
    mlflow,
    np,
    r2_score,
    y_test,
    y_train,
):
    if X_train is not None:
        # Iniciar un nuevo "run" (experimento) en MLflow
        with mlflow.start_run(run_name="RandomForest_Base") as run:

            # --- 1. Definir Modelo y Parámetros ---
            # Estos son los "ajustes" del modelo que queremos probar
            params = {
                "n_estimators": 100,    # Número de "árboles" en el bosque
                "max_depth": 15,        # Profundidad máxima de cada árbol
                "n_jobs": -1,           # Usar todos los procesadores
                "random_state": 42
            }
            model = RandomForestRegressor(**params)

            # --- 2. Registrar Parámetros en MLflow ---
            mlflow.log_params(params)
            print("Registrando parámetros en MLflow...")

            # --- 3. Entrenar el Modelo ---
            print("Entrenando el modelo...")
            model.fit(X_train, y_train)

            # --- 4. Realizar Predicciones ---
            print("Realizando predicciones...")
            y_pred = model.predict(X_test)

            # --- 5. Evaluar y Registrar Métricas ---
            print("Calculando y registrando métricas...")
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)

            metrics = {
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "r2": r2
            }
            mlflow.log_metrics(metrics)

            # --- 6. Registrar el Modelo en MLflow ---
            print("Registrando el modelo en MLflow...")
            mlflow.sklearn.log_model(model, "model")

            print("\n--- ¡Entrenamiento completado y registrado! ---")
            print(f"  Run ID: {run.info.run_id}")
            print("\n  Métricas de Evaluación (Test Set):")
            print(f"  R² (R-squared):    {r2:.3f}")
            print(f"  MAE (Error Absoluto Medio): {mae:.3f}")
            print(f"  RMSE (Error Cuadrático Medio): {rmse:.3f}")

    else:
        print("No se puede entrenar el modelo, no hay datos.")
        model = None
        metrics = {}
        y_pred = None

    return model, run, y_pred


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Visualización de Resultados

    ¿Qué tan bueno fue el modelo?
    * **R² (Coeficiente de Determinación):** El mejor valor es 1.0. Un R² de 0.85
        significa que el modelo puede "explicar" el 85% de la variación del stock.
    * **MAE (Error Absoluto Medio):** Nos dice, en promedio, por cuántas *unidades*
        se equivoca nuestra predicción. Si el MAE es 5, significa que nuestras
        predicciones están erradas por +/- 5 unidades en promedio.
    """)
    return


@app.cell
def _(model, pd, plt, sns, y_pred, y_test):
    if model is not None:
        # Crear un DataFrame para la visualización
        results_df = pd.DataFrame({'Real': y_test, 'Predicción': y_pred})

        plt.figure(figsize=(10, 6))

        # Scatter plot de Real vs Predicción
        sns.scatterplot(x='Real', y='Predicción', data=results_df, alpha=0.6, s=50)

        # Línea de 45 grados (predicción perfecta)
        max_val = max(y_test.max(), y_pred.max())
        min_val = min(y_test.min(), y_pred.min())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Predicción Perfecta')

        plt.title('Valores Reales vs. Predicciones del Modelo', fontsize=16, fontweight='bold')
        plt.xlabel('Stock Disponible Real (y_test)', fontsize=12)
        plt.ylabel('Stock Disponible Predicho (y_pred)', fontsize=12)
        plt.legend()
        plt.grid(True)

        # Guardar la figura
        plot_path = "prediccion_vs_real.png"
        plt.savefig(plot_path)
        plt.show()

        print(f"Gráfico guardado como '{plot_path}'")
    else:
        print("No se puede graficar, el modelo no fue entrenado.")

    return (plot_path,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Importancia de Features

    Esta es la parte más valiosa para el negocio.
    ¿Qué variables (features) usó más el modelo para tomar sus decisiones?

    Esto nos dirá si `lag_1`, `rotacion_estimada_30d` o `es_feriado`
    fueron las "pistas" más importantes.
    """)
    return


@app.cell
def _(X, mlflow, model, pd, plot_path, plt, run, sns):
    if model is not None:
        # 1. Obtener la importancia de los features del modelo
        importances = model.feature_importances_
        feature_names = X.columns

        # 2. Crear un DataFrame para visualizarlos
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        # 3. Visualizar el Top 20
        plt.figure(figsize=(12, 10))
        sns.barplot(
            x='Importance',
            y='Feature',
            data=feature_importance_df.head(20) # Mostrar solo el Top 20
        )
        plt.title('Top 20 Features más Importantes', fontsize=16, fontweight='bold')
        plt.xlabel('Importancia (calculada por RandomForest)', fontsize=12)
        plt.ylabel('Feature', fontsize=12)

        # Guardar la figura
        importance_plot_path = "feature_importance.png"
        plt.savefig(importance_plot_path)
        plt.show()

        # --- 4. Registrar gráficos en MLflow ---
        # Ahora que los gráficos están guardados, los subimos a MLflow
        # para que queden guardados junto al 'run'
        with mlflow.start_run(run_id=run.info.run_id):
             mlflow.log_artifact(plot_path)
             mlflow.log_artifact(importance_plot_path)

        print("Gráfico de importancia guardado y registrado en MLflow.")

    else:
        print("No se puede graficar, el modelo no fue entrenado.")
        feature_importance_df = pd.DataFrame()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Guardado del Modelo (Método Joblib)

    MLflow ya guardó el modelo por nosotros (en la carpeta `mlruns`).
    Sin embargo, para seguir tu ejemplo anterior, también lo guardaremos
    directamente en nuestra carpeta principal usando `joblib`.
    """)
    return


@app.cell
def _(joblib, model):
    if model is not None:
        model_filename = "modelo_stock_rfr.joblib"
        joblib.dump(model, model_filename)
        print(f"Modelo guardado exitosamente como '{model_filename}'")
    else:
        print("No hay modelo para guardar.")

    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9. ¡Inicia la UI de MLflow!

    ¡Todo está registrado! Para ver el dashboard con tus resultados:

    1.  Abre una **nueva terminal** (o Anaconda Prompt).
    2.  Navega (`cd`) a la carpeta de tu proyecto (donde está este cuaderno
        y la nueva carpeta `mlruns`).
    3.  Ejecuta el siguiente comando:

    ```bash
    mlflow ui
    ```

    4.  Abre tu navegador web y ve a la dirección que te indica
        (usualmente `http://127.0.0.1:5000`).
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
