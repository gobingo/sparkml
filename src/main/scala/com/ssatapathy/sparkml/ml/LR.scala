package com.ssatapathy.sparkml.ml

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel, OneVsRest, OneVsRestModel}
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.util.Random

class LR(private val spark: SparkSession) {

  import spark.implicits._

  def buildAndValidateLogisticRegression(trainData: DataFrame, testData: DataFrame, label: String): Unit = {
    val columns = trainData.columns.filter(!_.equals(label))

    val assembler = new VectorAssembler()
      .setInputCols(columns)
      .setOutputCol("features")

    val classifier = new LogisticRegression()
      .setFeaturesCol("features")
      .setLabelCol(label)
      .setPredictionCol("prediction")
      .setMaxIter(10)
      .setRegParam(0.1)
      .setElasticNetParam(0.5)

    val pipeline = new Pipeline().setStages(Array(assembler, classifier))

    val eval = new MulticlassClassificationEvaluator()
      .setLabelCol(label)
      .setPredictionCol("prediction")

    val model = pipeline.fit(trainData)

    val lrModel = model.asInstanceOf[PipelineModel].stages.last.asInstanceOf[LogisticRegressionModel]
    println(lrModel.extractParamMap())
    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

    val predictions = model.transform(testData)
    predictions.select(label, "prediction").show()

    val accuracy = eval.setMetricName("accuracy").evaluate(predictions)
    val precision = eval.setMetricName("weightedPrecision").evaluate(predictions)
    val recall = eval.setMetricName("weightedRecall").evaluate(predictions)
    val f1Score = eval.setMetricName("f1").evaluate(predictions)

    println("accuracy: " + accuracy)
    println("precision: " + precision)
    println("recall: " + recall)
    println("f1Score: " + f1Score)
  }

  def buildAndValidateLinearRegression(trainData: DataFrame, testData: DataFrame, label: String): Unit = {
    val columns = trainData.columns.filter(!_.equals(label))

    val classifier = new LinearRegression()
      .setFeaturesCol("indexedFeatures")
      .setLabelCol(label)
      .setPredictionCol("prediction")
      .setMaxIter(10)
      .setRegParam(0.1)
      .setElasticNetParam(0.5)

    val eval = new RegressionEvaluator()
      .setLabelCol(label)
      .setPredictionCol("prediction")

    val pipeline = new Pipeline().setStages(Array(classifier))

    val model = pipeline.fit(trainData)

    val lrModel = model.asInstanceOf[PipelineModel].stages.last.asInstanceOf[LinearRegressionModel]
    println(lrModel.extractParamMap())
    println(s"Coefficients: ${lrModel.coefficients}")

    val predictions = model.transform(testData)
    predictions.select(label, "prediction").show()

    val mse = eval.setMetricName("mse").evaluate(predictions)
    val rmse = eval.setMetricName("rmse").evaluate(predictions)
    val r2 = eval.setMetricName("r2").evaluate(predictions)
    val mae = eval.setMetricName("mae").evaluate(predictions)

    println("mse: " + rmse)
    println("rmse: " + rmse)
    println("r2: " + r2)
    println("mae: " + mae)
  }

  def buildValidateAndSaveLogisticRegression(trainData: DataFrame, testData: DataFrame, label: String): Unit = {
    val columns = trainData.columns.filter(!_.equals(label))

    val assembler = new VectorAssembler()
      .setInputCols(columns)
      .setOutputCol("features")

    val classifier = new LogisticRegression()
      .setFeaturesCol("features")
      .setLabelCol(label)
      .setPredictionCol("prediction")
      .setMaxIter(20)
      .setRegParam(0.01)
      .setElasticNetParam(0.5)

    val pipeline = new Pipeline().setStages(Array(assembler, classifier))

    val eval = new MulticlassClassificationEvaluator()
      .setLabelCol(label)
      .setPredictionCol("prediction")

    val model = pipeline.fit(trainData)

    val lrModel = model.asInstanceOf[PipelineModel].stages.last.asInstanceOf[LogisticRegressionModel]
    println(lrModel.extractParamMap())
    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

    val predictions = model.transform(testData)
    predictions.select(label, "prediction").show()

    val accuracy = eval.setMetricName("accuracy").evaluate(predictions)
    val precision = eval.setMetricName("weightedPrecision").evaluate(predictions)
    val recall = eval.setMetricName("weightedRecall").evaluate(predictions)
    val f1Score = eval.setMetricName("f1").evaluate(predictions)

    println("accuracy: " + accuracy)
    println("precision: " + precision)
    println("recall: " + recall)
    println("f1Score: " + f1Score)

    model.write.overwrite().save("/tmp/spark_lr_model")
  }

  def buildAndValidateLogisticRegressionWithValidationSet(
    trainData: DataFrame, testData: DataFrame, label: String): Unit = {

    val columns = trainData.columns.filter(!_.equals(label))

    val assembler = new VectorAssembler()
      .setInputCols(columns)
      .setOutputCol("features")

    val classifier = new LogisticRegression()
      .setFeaturesCol("features")
      .setLabelCol(label)
      .setPredictionCol("prediction")

    val eval = new MulticlassClassificationEvaluator()
      .setLabelCol(label)
      .setPredictionCol("prediction")

    val pipeline = new Pipeline().setStages(Array(assembler, classifier))

    val gridSearch = new ParamGridBuilder()
      .addGrid(classifier.maxIter, Array(10, 20))
      .addGrid(classifier.regParam, Array(0.01, 0.1))
      .addGrid(classifier.elasticNetParam, Array(0.05, 0.5))
      .addGrid(classifier.fitIntercept)
      .build()

    val tvs = new TrainValidationSplit()
      .setSeed(Random.nextLong())
      .setEstimator(pipeline)
      .setEstimatorParamMaps(gridSearch)
      .setEvaluator(eval)
      .setTrainRatio(0.8)

    val model = tvs.fit(trainData).bestModel

    val lrModel = model.asInstanceOf[PipelineModel].stages.last.asInstanceOf[LogisticRegressionModel]
    println(lrModel.extractParamMap())
    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

    val predictions = model.transform(testData)
    predictions.select(label, "prediction").show()

    val accuracy = eval.setMetricName("accuracy").evaluate(predictions)
    val precision = eval.setMetricName("weightedPrecision").evaluate(predictions)
    val recall = eval.setMetricName("weightedRecall").evaluate(predictions)
    val f1Score = eval.setMetricName("f1").evaluate(predictions)

    println("accuracy: " + accuracy)
    println("precision: " + precision)
    println("recall: " + recall)
    println("f1Score: " + f1Score)
  }

  def buildAndValidateLogisticRegressionWithCV(trainData: DataFrame, testData: DataFrame, label: String): Unit = {
    val columns = trainData.columns.filter(!_.equals(label))

    val assembler = new VectorAssembler()
      .setInputCols(columns)
      .setOutputCol("features")

    val classifier = new LogisticRegression()
      .setFeaturesCol("features")
      .setLabelCol(label)
      .setPredictionCol("prediction")

    val eval = new MulticlassClassificationEvaluator()
      .setLabelCol(label)
      .setPredictionCol("prediction")

    val gridSearch = new ParamGridBuilder()
      .addGrid(classifier.fitIntercept)
      .addGrid(classifier.maxIter, Array(10, 20))
      .addGrid(classifier.regParam, Array(0.01, 0.1))
      .addGrid(classifier.elasticNetParam, Array(0.05, 0.5))
      .build()

    val pipeline = new Pipeline().setStages(Array(assembler, classifier))

    val cv = new CrossValidator()
      .setSeed(Random.nextLong())
      .setEstimator(pipeline)
      .setEstimatorParamMaps(gridSearch)
      .setEvaluator(eval)
      .setNumFolds(5)

    val model = cv.fit(trainData).bestModel

    val lrModel = model.asInstanceOf[PipelineModel].stages.last.asInstanceOf[LogisticRegressionModel]
    println(lrModel.extractParamMap())
    println(s"coefficients: ${lrModel.coefficients}")

    val predictions = model.transform(testData)
    predictions.select(label, "prediction").show()

    val accuracy = eval.setMetricName("accuracy").evaluate(predictions)
    val precision = eval.setMetricName("weightedPrecision").evaluate(predictions)
    val recall = eval.setMetricName("weightedRecall").evaluate(predictions)
    val f1Score = eval.setMetricName("f1").evaluate(predictions)

    println("accuracy: " + accuracy)
    println("precision: " + precision)
    println("recall: " + recall)
    println("f1Score: " + f1Score)
  }

  def applyLR(): Unit = {
    val model = PipelineModel.load("/tmp/spark_lr_model")

    val test = spark.createDataFrame(Seq(
      (3, 150, 65, 27, 0, 32, 0.4, 50)
    )).toDF("Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
      "DiabetesPedigreeFunction", "Age")

    model.transform(test).select("prediction").show()
  }

  def buildAndValidateMulticlassClassifier(trainData: DataFrame, testData: DataFrame, label: String): Unit = {
    val columns = trainData.columns.filter(!_.contains(label))

    val assembler = new VectorAssembler()
      .setInputCols(columns)
      .setOutputCol("features")

    val classifier = new LogisticRegression()
      .setFeaturesCol("features")
      .setLabelCol(label)
      .setPredictionCol("prediction")
      .setMaxIter(20)
      .setRegParam(0.1)
      .setElasticNetParam(0.5)
      .setFitIntercept(true)

    val eval = new MulticlassClassificationEvaluator()
      .setLabelCol(label)
      .setPredictionCol("prediction")

    val ovr = new OneVsRest()
      .setClassifier(classifier)
      .setFeaturesCol("features")
      .setLabelCol(label)
      .setPredictionCol("prediction")

    val pipeline = new Pipeline().setStages(Array(assembler, ovr))

    val model = pipeline.fit(trainData)

    val ovrModel = model.asInstanceOf[PipelineModel].stages.last.asInstanceOf[OneVsRestModel]
    println(ovrModel.extractParamMap())

    val predictions = model.transform(testData)

    val accuracy = eval.setMetricName("accuracy").evaluate(predictions)
    val precision = eval.setMetricName("weightedPrecision").evaluate(predictions)
    val recall = eval.setMetricName("weightedRecall").evaluate(predictions)
    val f1Score = eval.setMetricName("f1").evaluate(predictions)

    println("accuracy: " + accuracy)
    println("precision: " + precision)
    println("recall: " + recall)
    println("f1Score: " + f1Score)
  }

}
