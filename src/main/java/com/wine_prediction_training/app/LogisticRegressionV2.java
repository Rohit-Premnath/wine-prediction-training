package com.wine_prediction_training.app;

import java.io.IOException;
import org.apache.spark.sql.Row;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.LogisticRegressionTrainingSummary;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.SparkSession;
import org.apache.log4j.Logger;
import org.apache.log4j.LogManager;
import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.split;
import static org.apache.spark.sql.functions.*;

public class LogisticRegressionV2 {
    public static final Logger logger = LogManager.getLogger(App.class);
    private static final String TRAINING_DATASET = "s3a://winepredictionrp/TrainingDataset.csv";
    private static final String VALIDATION_DATASET = "s3a://winepredictionrp/ValidationDataset.csv";
    private static final String MODEL_PATH = "s3a://winepredictionrp/LogisticRegressionModel";

    public void trainModel(SparkSession spark) {
        Dataset<Row> lblFeatureDf = getDataFrame(spark, true, TRAINING_DATASET).cache();
        LogisticRegression logReg = new LogisticRegression().setMaxIter(100).setRegParam(0.0);

        Pipeline pl1 = new Pipeline();
        pl1.setStages(new PipelineStage[] { logReg });

        PipelineModel model1 = pl1.fit(lblFeatureDf);

        LogisticRegressionModel lrModel = (LogisticRegressionModel) (model1.stages()[0]);
        LogisticRegressionTrainingSummary trainingSummary = lrModel.summary();
        double accuracy = trainingSummary.accuracy();
        double falsePositiveRate = trainingSummary.weightedFalsePositiveRate();
        double truePositiveRate = trainingSummary.weightedTruePositiveRate();
        double fMeasure = trainingSummary.weightedFMeasure();
        double precision = trainingSummary.weightedPrecision();
        double recall = trainingSummary.weightedRecall();

        System.out.println();
        System.out.println("Training DataSet Metrics ");

        System.out.println("Accuracy: " + accuracy);
        System.out.println("FPR: " + falsePositiveRate);
        System.out.println("TPR: " + truePositiveRate);
        System.out.println("F-measure: " + fMeasure);
        System.out.println("Precision: " + precision);
        System.out.println("Recall: " + recall);

        Dataset<Row> testingDf1 = getDataFrame(spark, true, VALIDATION_DATASET).cache();

        Dataset<Row> results = model1.transform(testingDf1);

        System.out.println("\n Validation Training Set Metrics");
        results.select("features", "label", "prediction").show(5, false);
        printMertics(results);

        try {
            model1.write().overwrite().save(MODEL_PATH);
        } catch (IOException e) {
            // System.out.print(e.printStackTrace());
            e.printStackTrace();
        }
    }

    private void printMertics(Dataset<Row> predictions) {
        System.out.println();
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator();
        evaluator.setMetricName("accuracy");
        System.out.println("The accuracy of the model is " + evaluator.evaluate(predictions));

        evaluator.setMetricName("accuracy");
        double accuracy1 = evaluator.evaluate(predictions);
        System.out.println("Test Error = " + (1.0 - accuracy1));

        evaluator.setMetricName("f1");
        double f1 = evaluator.evaluate(predictions);

        evaluator.setMetricName("weightedPrecision");
        double weightedPrecision = evaluator.evaluate(predictions);

        evaluator.setMetricName("weightedRecall");
        double weightedRecall = evaluator.evaluate(predictions);

        System.out.println("===============================Performance Matrix ====================================");
        System.out.println("Accuracy: " + accuracy1);
        System.out.println("F1: " + f1);
        System.out.println("===============================Performance Matrix Ends====================================");
        // System.out.println("Precision: " + weightedPrecision);
        // System.out.println("Recall: " + weightedRecall);
    }

    private Dataset<Row> getDataFrame(SparkSession spark, boolean transform, String name) {

        System.out.println("here1");
        Dataset<Row> df = spark.read().format("csv").option("header", "true").option("multiline", "true")
                .option("sep", ";").option("quote", "\"").option("escape","\"")
                .option("dateFormat", "M/d/y").option("inferSchema", true).load(name);

        Dataset<Row> validationDf = df.withColumnRenamed("fixed acidity", "fixed_acidity")

                .withColumnRenamed("volatile acidity", "volatile_acidity")
                .withColumnRenamed("citric acid", "citric_acid")
                .withColumnRenamed("residual sugar", "residual_sugar")
                .withColumnRenamed("chlorides", "chlorides")
                .withColumnRenamed("free sulfur dioxide", "free_sulfur_dioxide")
                .withColumnRenamed("total sulfur dioxide", "total_sulfur_dioxide")
                .withColumnRenamed("density", "density")
                .withColumnRenamed("pH", "pH")
                .withColumnRenamed("sulphates", "sulphates")
                .withColumnRenamed("alcohol", "alcohol")
                .withColumnRenamed("quality", "label");
        df.show(5, false);

    // Now split _c0 column manually
    // Dataset<Row> splitDf = df.select(
    //     split(col("_c0"), ";").getItem(0).cast("double").alias("fixed_acidity"),
    //     split(col("_c0"), ";").getItem(1).cast("double").alias("volatile_acidity"),
    //     split(col("_c0"), ";").getItem(2).cast("double").alias("citric_acid"),
    //     split(col("_c0"), ";").getItem(3).cast("double").alias("residual_sugar"),
    //     split(col("_c0"), ";").getItem(4).cast("double").alias("chlorides"),
    //     split(col("_c0"), ";").getItem(5).cast("double").alias("free_sulfur_dioxide"),
    //     split(col("_c0"), ";").getItem(6).cast("double").alias("total_sulfur_dioxide"),
    //     split(col("_c0"), ";").getItem(7).cast("double").alias("density"),
    //     split(col("_c0"), ";").getItem(8).cast("double").alias("pH"),
    //     split(col("_c0"), ";").getItem(9).cast("double").alias("sulphates"),
    //     split(col("_c0"), ";").getItem(10).cast("double").alias("alcohol"),
    //     split(col("_c0"), ";").getItem(11).cast("double").alias("label")   
    // );

    validationDf.show(5, false);
    System.out.println("Split into proper columns");


        Dataset<Row> lblFeatureDf = validationDf.select("label", "alcohol", "sulphates", "pH",
                "density", "free_sulfur_dioxide", "total_sulfur_dioxide", "chlorides", "residual_sugar",
                "citric_acid", "volatile_acidity", "fixed_acidity");

        lblFeatureDf = lblFeatureDf.na().drop().cache();

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[] { "alcohol", "sulphates", "pH", "density",
                        "free_sulfur_dioxide", "total_sulfur_dioxide", "chlorides", "residual_sugar",
                        "citric_acid", "volatile_acidity", "fixed_acidity" })
                .setOutputCol("features");

        if (transform)
            lblFeatureDf = assembler.transform(lblFeatureDf).select("label", "features");

        return lblFeatureDf;
    }
}