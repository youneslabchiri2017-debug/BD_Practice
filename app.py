import sys
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator

def main():
    # 0. Verificar que se ha pasado la ruta de los datos como argumento
    if len(sys.argv) < 2:
        print("Uso: spark-submit predict_app.py <ruta_de_datos_test>")
        sys.exit(-1)

    data_path = sys.argv[1]

    # Inicializar la sesión de Spark
    spark = SparkSession.builder \
        .appName("Flight Delay Prediction Application") \
        .getOrCreate()

    print(f"Cargando datos desde: {data_path}")

    try:
        # 1. Load the test data from the location passed
        # Nota: Asegúrate de que el formato (csv, parquet, etc.) coincida con tus datos
        test_data = spark.read.csv(data_path, header=True, inferSchema=True)

        # 2. Load your best_model
        # Cargamos el modelo que guardamos previamente como PipelineModel
        model_path = "best_model"
        print(f"Cargando modelo desde: {model_path}")
        best_model = PipelineModel.load(model_path)

        # 3. Perform some predictions on the test data
        print("Realizando predicciones...")
        predictions = best_model.transform(test_data)

        # Mostrar algunas predicciones (opcional, para verificar)
        predictions.select("ArrDelay", "prediction").show(10)

        # 4. Perform a complete performance evaluation on the test data
        print("Evaluando rendimiento final...")
        
        evaluador_rmse = RegressionEvaluator(labelCol="ArrDelay", predictionCol="prediction", metricName="rmse")
        evaluador_mae = RegressionEvaluator(labelCol="ArrDelay", predictionCol="prediction", metricName="mae")
        evaluador_r2 = RegressionEvaluator(labelCol="ArrDelay", predictionCol="prediction", metricName="r2")

        rmse = evaluador_rmse.evaluate(predictions)
        mae = evaluador_mae.evaluate(predictions)
        r2 = evaluador_r2.evaluate(predictions)

        print("-" * 30)
        print(f"MÉTRICAS DE EVALUACIÓN FINAL:")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print(f"R2:   {r2:.4f}")
        print("-" * 30)

    except Exception as e:
        print(f"Error durante la ejecución: {e}")
    finally:
        spark.stop()

if __name__ == "__main__":
    main()