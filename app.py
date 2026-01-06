import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator

def main():
    # Check for arguments
    if len(sys.argv) < 2:
        print("Usage: spark-submit app.py <data_route>", file=f)
        sys.exit(-1)

    data_path = sys.argv[1]

    # Initialize spark app
    spark = SparkSession.builder \
        .appName("Flight Delay Prediction App") \
        .getOrCreate()

    print(f"Loading data from: {data_path}")
    with open("output.txt", "w") as f:

        try:
            # Load test data 
            df = spark.read.csv(
                data_path,
                header=True,
                inferSchema=True,
                nullValue="NA"
            )

            
            # Load model
            model_path = "best_model"
            print(f"Loading model from: {model_path}")
            best_model = PipelineModel.load(model_path)

            # Perform some predictions on the test data
            print("Predicting...", file=f)
            predictions = best_model.transform(df)

            # Show some result
            rows = predictions.select("ArrDelay", "prediction").take(10)

            print("\nPredictions sample:", file=f)
            print("-" * 40, file=f)
            for r in rows:
                print(f"ArrDelay real: {r['ArrDelay']}, Prediction: {r['prediction']:.2f}", file=f)
            print("-" * 40, file=f)


            # Perform a complete performance evaluation on the test data
            print("Evaluating performance...", file=f)
            
            evaluador_rmse = RegressionEvaluator(labelCol="ArrDelay", predictionCol="prediction", metricName="rmse")
            evaluador_mae = RegressionEvaluator(labelCol="ArrDelay", predictionCol="prediction", metricName="mae")
            evaluador_r2 = RegressionEvaluator(labelCol="ArrDelay", predictionCol="prediction", metricName="r2")

            rmse = evaluador_rmse.evaluate(predictions)
            mae = evaluador_mae.evaluate(predictions)
            r2 = evaluador_r2.evaluate(predictions)

            print("\n" + "="*40)
            print("FINAL PERFORMANCE REPORT", file=f)
            print("="*40, file=f)
            print(f"RMSE:  {rmse:.4f}", file=f)
            print(f"MAE:    {mae:.4f}", file=f)
            print(f"R2: {r2:.4f}", file=f)
            print("="*40, file=f)


            predictions.write.mode("overwrite").parquet("predictions_full_parquet")


        except Exception as e:
            print(f"ERROR: Couldn't processs model or data: {e}", file=f)
        finally:
            spark.stop()

if __name__ == "__main__":
    main()