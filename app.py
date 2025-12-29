import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator

def main():
    # 0. Verificar que se ha pasado la ruta de los datos como argumento
    if len(sys.argv) < 2:
        print("Uso: spark-submit predict_app.py <ruta_de_datos_test>", file=f)
        sys.exit(-1)

    data_path = sys.argv[1]

    # Inicializar la sesión de Spark
    spark = SparkSession.builder \
        .appName("Flight Delay Prediction - Production App") \
        .getOrCreate()

    print(f"Cargando datos desde: {data_path}")
    with open("salida.txt", "w") as f:

        try:
            # 1. Load the test data from the location passed
            df_raw = spark.read.csv(data_path, header=True, inferSchema=True)

            # --- LÓGICA DE FILTRADO (Esencial antes de la predicción) ---
            print("Limpiando y filtrando datos de entrada...", file=f)
            
            # Eliminar vuelos cancelados
            df = df_raw.filter(col("Cancelled") != 1)

            # Eliminar nulos en columnas críticas de tiempo
            df = df.dropna(subset=["CRSDepTime", "CRSArrTime"])

            # Eliminar nulos en la variable objetivo (necesario para la evaluación final)
            df = df.na.drop(subset=["ArrDelay"])
            
            print(f"Registros tras limpieza: {df.count()}", file=f)
            # 2. Load your best_model
            # Cargamos el modelo que guardamos previamente como PipelineModel
            model_path = "best_model"
            print(f"Cargando modelo desde: {model_path}")
            best_model = PipelineModel.load(model_path)

            # 3. Perform some predictions on the test data
            print("Realizando predicciones...", file=f)
            predictions = best_model.transform(df)

            # Mostrar una muestra de los resultados
            predictions.select("FlightNum", "ArrDelay", "prediction").show(10)

            # 4. Perform a complete performance evaluation on the test data
            print("Evaluando rendimiento final...", file=f)
            
            evaluador_rmse = RegressionEvaluator(labelCol="ArrDelay", predictionCol="prediction", metricName="rmse")
            evaluador_mae = RegressionEvaluator(labelCol="ArrDelay", predictionCol="prediction", metricName="mae")
            evaluador_r2 = RegressionEvaluator(labelCol="ArrDelay", predictionCol="prediction", metricName="r2")

            rmse = evaluador_rmse.evaluate(predictions)
            mae = evaluador_mae.evaluate(predictions)
            r2 = evaluador_r2.evaluate(predictions)

            print("\n" + "="*40)
            print("INFORME DE RENDIMIENTO FINAL", file=f)
            print("="*40, file=f)
            print(f"RMSE (Error cuadrático):  {rmse:.4f}", file=f)
            print(f"MAE  (Error absoluto):    {mae:.4f}", file=f)
            print(f"R2   (Varianza explicada): {r2:.4f}", file=f)
            print("="*40, file=f)

        except Exception as e:
            print(f"ERROR: No se pudo procesar el modelo o los datos. Detalle: {e}", file=f)
        finally:
            spark.stop()

if __name__ == "__main__":
    main()