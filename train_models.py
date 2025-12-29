# %% [markdown]
# # Spark Practical Work
# 
# Authors:
#  - Ahajjan Ziggaf Kanjaa, Mohammed
#  - Labchiri Boukhalef, Younes
#  - Ramírez Castaño, Víctor

# %% [markdown]
# ### Data loading

# %%
from pyspark.sql import SparkSession
from pyspark.sql.functions import (col, sum)
from pyspark.sql.types import StringType
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    SQLTransformer,
    Imputer,
    StringIndexer, 
    OneHotEncoder, 
    VectorAssembler, 
    StandardScaler
)
from pyspark.ml.regression import (
    DecisionTreeRegressor,
    RandomForestRegressor,
    GBTRegressor
)

spark = SparkSession.builder.config("spark.driver.memory", "8g").appName("FlightModelPrediction").getOrCreate()

data_path = "../training_data/flight_data/1988.csv"

df = spark.read.csv(
    data_path,
    header=True,
    inferSchema=True,
    nullValue="NA"
)

# %% [markdown]
# ### Explaratory data analysis (EDA)

# %%
#Drop the variables that contain information that is unknown at the time the plane takes off

columns_to_drop = [
    "ArrTime", "ActualElapsedTime", "AirTime", "TaxiIn",
    "Diverted", "CarrierDelay", "WeatherDelay",
    "NASDelay", "SecurityDelay", "LateAircraftDelay"
]

df.printSchema()
df.describe().show()

df.select([
    sum(col(c).isNull().cast("int")).alias(c)
    for c in df.columns
]).show(truncate=False)

# %% [markdown]
# ### Feature selection (FSS)

# %%
#A cancelled flight is not considered a delay, so that it does not give us useful information.
df = df.filter(col("Cancelled") != 1)

#If we do not have the attributes 'CRSDepTime' or 'CRSArrTime' we delete that instance
df = df.dropna(subset=["CRSDepTime", "CRSArrTime"])

df = df.na.drop(subset=["ArrDelay"])

df.select([
    sum(col(c).isNull().cast("int")).alias(c)
    for c in df.columns
]).show(truncate=False)

df = df.withColumn("TaxiOut", col("TaxiOut").cast("int"))

train_data, test_data = df.randomSplit([0.8, 0.2], seed=89)


# %%
# --- Construcción del Pipeline ---
stages = []

# %% [markdown]
# ### New variables

# %%
sql_logic = """
SELECT 
    Year, Month, DayofMonth, DayOfWeek, DepTime, CRSDepTime, CRSArrTime, UniqueCarrier, CRSElapsedTime, ArrDelay, DepDelay, Origin, Dest, Distance,
    -- 1. Calculamos la mediana dinámicamente y si es NULL (columna vacía) ponemos 0
    COALESCE(
        (SELECT percentile_approx(TaxiOut, 0.5) FROM __THIS__), 
        0
    ) AS TaxiOut,

    -- 2. Cálculo de TakeOffTime usando la columna limpia anterior
    CAST(date_format(
        to_timestamp(concat(Year, lpad(Month, 2, '0'), lpad(DayofMonth, 2, '0'), lpad(CRSDepTime, 4, '0')), 'yyyyMMddHHmm') 
        + (COALESCE(CAST(DepDelay AS INT), 0) + COALESCE(CAST(TaxiOut AS INT), 0)) * INTERVAL 1 MINUTE, 
    'HHmm') AS INT) as TakeOffTime,
    
    -- 3. Cálculo de LandingEst
    CAST(date_format(
        to_timestamp(concat(Year, lpad(Month, 2, '0'), lpad(DayofMonth, 2, '0'), lpad(CRSDepTime, 4, '0')), 'yyyyMMddHHmm') 
        + COALESCE(CAST(CRSElapsedTime AS INT), 0) * INTERVAL 1 MINUTE, 
    'HHmm') AS INT) as LandingEst
FROM __THIS__
"""

# Creamos el transformador
sql_trans = SQLTransformer(statement=sql_logic)

stages.append(sql_trans)

# %%
# --- 1. Definición de Variables ---

# Selecciona tus 5 variables categóricas (Ejemplos basados en flight data)
cat_cols = ["UniqueCarrier", "Origin", "Dest", "Month", "DayOfWeek"]

# Selecciona tus 10 variables numéricas
num_cols = [
    "DepDelay", "TaxiOut", "Distance", "CRSElapsedTime", 
    "LandingEst", "TakeOffTime", "DayofMonth", 
    "Year", "CRSDepTime", "CRSArrTime"
]


# A) Procesamiento de Categóricas (StringIndexer + OHE)
input_cols_ohe = []

for c in cat_cols:
    # 1. Indexar: Convierte strings a índices numéricos
    indexer = StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
    
    # 2. OHE: Convierte índices a vectores dispersos (sparse vectors)
    encoder = OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_ohe")
    
    # Añadimos pasos al pipeline y guardamos el nombre de la columna de salida
    stages += [indexer, encoder]
    input_cols_ohe.append(f"{c}_ohe")

# B) Procesamiento de Numéricas (Assembler + StandardScaler)
# Nota: StandardScaler en Spark requiere una columna de tipo Vector, no columnas sueltas.

# 1. Agrupar todas las numéricas en un solo vector temporal
num_assembler = VectorAssembler(inputCols=num_cols, outputCol="num_features_raw", handleInvalid="skip")
stages.append(num_assembler)

# 2. Estandarizar ese vector (Media 0, Desviación Estándar 1)
scaler = StandardScaler(
    inputCol="num_features_raw", 
    outputCol="num_features_scaled", 
    withStd=True, 
    withMean=True
)
stages.append(scaler)

# C) Ensamblaje Final (Unir OHE + Numéricas Escaladas)

# Juntamos las columnas OHE y la columna de numéricas ya escaladas
assembler_all_inputs = input_cols_ohe + ["num_features_scaled"]

assembler_all = VectorAssembler(
    inputCols=assembler_all_inputs, 
    outputCol="features" # Esta es la columna estándar para modelos ML en Spark
)
stages.append(assembler_all)


# %% [markdown]
# ### Baseline model training
# 
# Three baseline models are trained: `LogisticRegression(max_iter=1000)`, `DecisionTreeClassifier()`, and `MLPClassifier(max_iter=500)`. These models are fitted on the feature-selected and scaled training data. Training multiple models at this stage establishes performance benchmarks for comparison and identifies which algorithms are initially more suitable for the dataset before any tuning is applied.

# %%
from pyspark.ml import Pipeline
from pyspark.ml.regression import DecisionTreeRegressor, RandomForestRegressor, GBTRegressor
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator

# 1. Definición de Modelos
dt = DecisionTreeRegressor(labelCol="ArrDelay", featuresCol="features")
rf = RandomForestRegressor(labelCol="ArrDelay", featuresCol="features")
gbt = GBTRegressor(labelCol="ArrDelay", featuresCol="features")

# Creación de Pipelines (usando concatenación para mayor limpieza)
pipeline_dt = Pipeline(stages=stages + [dt])
pipeline_rf = Pipeline(stages=stages + [rf])
pipeline_gbt = Pipeline(stages=stages + [gbt])

# 2. Rejillas de parámetros (ParamGrids)
# IMPORTANTE: Los parámetros deben coincidir con la instancia del modelo
'''
paramGrid_dt = (ParamGridBuilder()
    .addGrid(dt.maxDepth, [5, 10])
    .addGrid(dt.maxBins, [32, 48]) # DT no usa regParam, usamos maxBins
    .build())

paramGrid_rf = (ParamGridBuilder()
    .addGrid(rf.numTrees, [20, 50]) # Parámetro específico de RF
    .addGrid(rf.maxDepth, [5, 10])
    .build())

paramGrid_gbt = (ParamGridBuilder()
    .addGrid(gbt.maxIter, [10, 20]) # Parámetro específico de GBT (iteraciones)
    .addGrid(gbt.maxDepth, [3, 5])
    .build())
'''

paramGrid_dt = (ParamGridBuilder()
    .addGrid(dt.maxDepth, [5])    # Puedes añadir mas valores si tienes el musculo tecnico para hacerlo. Nosotros no lo tenemos
    .addGrid(dt.maxBins, [32])   # Puedes añadir mas valores si tienes el musculo tecnico para hacerlo. Nosotros no lo tenemos
    .build())

paramGrid_rf = (ParamGridBuilder()
    .addGrid(rf.numTrees, [20])   # Puedes añadir mas valores si tienes el musculo tecnico para hacerlo. Nosotros no lo tenemos
    .addGrid(rf.maxDepth, [5])    # Puedes añadir mas valores si tienes el musculo tecnico para hacerlo. Nosotros no lo tenemos
    .build())

paramGrid_gbt = (ParamGridBuilder()
    .addGrid(gbt.maxIter, [10])   # Puedes añadir mas valores si tienes el musculo tecnico para hacerlo. Nosotros no lo tenemos
    .addGrid(gbt.maxDepth, [3])    # Puedes añadir mas valores si tienes el musculo tecnico para hacerlo. Nosotros no lo tenemos
    .build())
# 3. CrossValidators
# Configuramos el evaluador una sola vez para reutilizarlo
evaluador = RegressionEvaluator(labelCol="ArrDelay", predictionCol="prediction", metricName="rmse")

cv_dt = CrossValidator(
    estimator=pipeline_dt, 
    estimatorParamMaps=paramGrid_dt,
    evaluator=evaluador,
    numFolds=3,
    seed=89,
    parallelism=1
)

cv_rf = CrossValidator(
    estimator=pipeline_rf,
    estimatorParamMaps=paramGrid_rf,
    evaluator=evaluador,
    numFolds=3,
    seed=89,
    parallelism=1
)

cv_gbt = CrossValidator(
    estimator=pipeline_gbt,
    estimatorParamMaps=paramGrid_gbt,
    evaluator=evaluador,
    numFolds=3,
    seed=89,
    parallelism=1
)




# %% [markdown]
# ### Baseline evaluation
# 
# The `evaluate_full()` function is used to evaluate each model on the test set. It calculates standard metrics including accuracy, precision, recall, and F1-score. It also plots a confusion matrix, the ROC curve with AUC, and the precision-recall curve with AUC-PR. These metrics provide a comprehensive overview of each model's performance and allow for a more nuanced understanding of strengths and weaknesses, especially in cases of class imbalance.

# %%


modelos_cv = [("Decision Tree", cv_dt), ("Random Forest", cv_rf), ("GBT", cv_gbt)]
resultados = []

# Evaluadores
eval_mae = RegressionEvaluator(labelCol="ArrDelay", metricName="mae")
# 'evaluador' ya lo tienes definido arriba como RMSE

for nombre, cv in modelos_cv:
    print(f"Iniciando Cross-Validation para {nombre}...")
    
    # 1. AJUSTE DE HIPERPARÁMETROS (Tuning)
    # Aquí Spark entrena los 'n' folds y selecciona la mejor combinación de la rejilla
    spark.catalog.clearCache()
    import gc
    gc.collect()
    cv_model = cv.fit(train_data) 
    
    # 2. EVALUACIÓN (Validación con datos no vistos)
    # Usamos el cv_model (que ya contiene el mejor modelo interno) para predecir
    predicciones = cv_model.transform(test_data)
    
    rmse = evaluador.evaluate(predicciones) # RMSE
    mae = eval_mae.evaluate(predicciones)   # MAE
    
    # Tu lógica de decisión: Si RMSE es muy alto respecto al MAE, priorizar MAE
    score = mae if rmse > (2 * mae) else rmse
    metrica_usada = "MAE" if rmse > (2 * mae) else "RMSE"
    
    print(f"Resultados {nombre} -> RMSE: {rmse:.2f}, MAE: {mae:.2f} (Score: {score:.2f} vía {metrica_usada})")
    
    # Guardamos todo el objeto cv_model para poder extraer el mejor modelo después
    resultados.append({
        "nombre": nombre,
        "modelo_fit": cv_model, 
        "rmse": rmse,
        "mae": mae,
        "score_final": score,
        "metrica": metrica_usada
    })

# --- PASO D: ENCONTRAR Y GUARDAR EL MEJOR ---

# Encontrar el mejor resultado basado en tu score_final
mejor_resultado = min(resultados, key=lambda x: x["score_final"])

print("-" * 30)
print(f"GANADOR: {mejor_resultado['nombre']}")
print(f"Criterio: {mejor_resultado['metrica']} de {mejor_resultado['score_final']:.4f}")
print("-" * 30)

# Extraer el modelo definitivo (el que mejor funcionó en el CV)
# Nota: cv_model.bestModel devuelve el PipelineModel con los mejores parámetros
mejor_modelo_final = mejor_resultado['modelo_fit'].bestModel

# 3. ALMACENAMIENTO DEL MODELO (Requisito del proyecto)
path_guardado = "best_model"
mejor_modelo_final.write().overwrite().save(path_guardado)

print(f"Éxito: El modelo '{mejor_resultado['nombre']}' ha sido guardado en la carpeta '{path_guardado}'")

'''modelos_cv = [("Decision Tree", cv_dt), ("Random Forest", cv_rf), ("GBT", cv_gbt)]
resultados = []

eval_mae = RegressionEvaluator(labelCol="ArrDelay", metricName="mae")

for nombre, cv in modelos_cv:
    print(f"Entrenando {nombre}...")
    fit_model = cv.fit(train_data) # train_data debe estar definido previamente
    predicciones = fit_model.transform(test_data)
    
    rmse = evaluador.evaluate(predicciones) # RMSE
    mae = eval_mae.evaluate(predicciones)   # MAE
    
    # Tu lógica: Si RMSE > 2*MAE, fijarse en MAE, si no en RMSE
    score = mae if rmse > (2 * mae) else rmse
    metrica_usada = "MAE" if rmse > (2 * mae) else "RMSE"
    
    resultados.append({
        "nombre": nombre,
        "rmse": rmse,
        "mae": mae,
        "score_final": score,
        "metrica": metrica_usada
    })

# Encontrar el mejor
mejor_modelo = min(resultados, key=lambda x: x["score_final"])
print(f"\nEl mejor modelo es {mejor_modelo['nombre']} basado en {mejor_modelo['metrica']}")'''

# %% [markdown]
# ### Hyperparameter tuning
# 
# GridSearchCV is used for hyperparameter optimization for each model: logistic regression (`C`, `penalty`, `solver`), decision tree (`max_depth`, `min_samples_split`), and neural network (`hidden_layer_sizes`, `alpha`). The code searches across multiple parameter combinations with 5-fold cross-validation to select the configuration that maximizes accuracy. This step is essential for improving model generalization and ensuring that the final models perform optimally.

# %% [markdown]
# ### Final evaluation
# 
# After tuning, the code retrieves the best model via `grid.best_estimator_` and evaluates it on the test set with `accuracy_score` and `classification_report` again. This step confirms how much the optimized model improves over the baseline. Comparing results allows the user to quantify performance gains resulting from hyperparameter optimization and feature selection.

# %% [markdown]
# ### Interpretability
# 
# For interpretability, the logistic regression coefficients (betas) are extracted from `best_lr.coef_` and plotted as a horizontal bar chart to visualize the influence of each feature. The decision tree is visualized using `plot_tree()` to show the full branching structure, feature thresholds, Gini impurity, and leaf outcomes. These visualizations help understand how the models make predictions and identify the most important features driving classification decisions

# %% [markdown]
# ### Comparison Before vs After Tuning
# 
# This section summarizes the key metrics (accuracy, precision, recall, F1-score) for all three models in a pandas DataFrame. It compares baseline and tuned models side by side, allowing a clear overview of the improvements achieved through feature selection and hyperparameter tuning. This comparative analysis helps identify which model benefits the most from tuning and provides actionable insights for model selection.


