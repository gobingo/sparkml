package com.ssatapathy.sparkml.ml

import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.{DataFrame, SparkSession}
import scala.util.Random

class GradientBoostedTrees(private val spark: SparkSession) {

  def buildAndValidateGBT(trainData: DataFrame, testData: DataFrame, label: String): Unit = {
    val columns = trainData.columns.filter(!_.equals(label))

    val assembler = new VectorAssembler()
      .setInputCols(columns)
      .setOutputCol("features")

    val classifier = new GBTClassifier()
      .setSeed(Random.nextLong())
      .setFeaturesCol("indexedFeatures")
      .setLabelCol(label)
      .setPredictionCol("prediction")
      .setImpurity("entropy")
      .setMaxDepth(20)
      .setMaxBins(20)
      .setMaxIter(10)

    val eval = new MulticlassClassificationEvaluator()
      .setLabelCol(label)
      .setPredictionCol("prediction")

    val pipeline = new Pipeline().setStages(Array(assembler, classifier))

    val model = pipeline.fit(trainData)

    val gbtModel = model.asInstanceOf[PipelineModel].stages.last.asInstanceOf[GBTClassificationModel]
    println(gbtModel.extractParamMap)

    gbtModel.featureImportances.toArray.zip(columns).sorted.reverse.foreach(println)

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

  def buildAndValidateGBTRegressor(trainData: DataFrame, testData: DataFrame, label: String): Unit = {
    val columns = trainData.columns.filter(!_.equals(label))
    columns.foreach(println)

    val classifier = new GBTRegressor()
      .setSeed(Random.nextLong())
      .setFeaturesCol("indexedFeatures")
      .setLabelCol(label)
      .setPredictionCol("prediction")
      .setImpurity("entropy")
      .setMaxDepth(20)
      .setMaxBins(328)
      .setMaxIter(10)

    val eval = new RegressionEvaluator()
      .setLabelCol(label)
      .setPredictionCol("prediction")

    val pipeline = new Pipeline().setStages(Array(classifier))

    val model = pipeline.fit(trainData)

    val gbtModel = model.asInstanceOf[PipelineModel].stages.last.asInstanceOf[GBTRegressionModel]
    println(gbtModel.extractParamMap)

    gbtModel.featureImportances.toArray.zip(columns).sorted.reverse.foreach(println)

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

  def buildAndValidateGBTWithValidationSet(trainData: DataFrame, testData: DataFrame, label: String): Unit = {
    val columns = trainData.columns.filter(!_.equals(label))
    columns.foreach(input => println(input))

    val assembler = new VectorAssembler()
      .setInputCols(columns)
      .setOutputCol("features")

    val classifier = new GBTClassifier()
      .setSeed(Random.nextLong())
      .setFeaturesCol("features")
      .setLabelCol(label)
      .setPredictionCol("prediction")

    val pipeline = new Pipeline().setStages(Array(assembler, classifier))

    val eval = new MulticlassClassificationEvaluator()
      .setLabelCol(label)
      .setPredictionCol("prediction")

    val gridSearch = new ParamGridBuilder()
      .addGrid(classifier.maxIter, Array(10, 20))
      .addGrid(classifier.maxBins, Array(10, 20))
      .addGrid(classifier.maxDepth, Array(10, 20))
      .build()

    val tvs = new TrainValidationSplit()
      .setSeed(Random.nextLong())
      .setEstimator(pipeline)
      .setEvaluator(eval)
      .setEstimatorParamMaps(gridSearch)
      .setTrainRatio(0.8)

    val model = tvs.fit(trainData).bestModel

    val gbtModel = model.asInstanceOf[PipelineModel].stages.last.asInstanceOf[GBTClassificationModel]

    println(gbtModel.extractParamMap)

    gbtModel.featureImportances.toArray.zip(columns).sorted.reverse.foreach(println)

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

  def buildAndValidateGBTWithGridSearchAndCV(trainData: DataFrame, testData: DataFrame, label: String): Unit = {
    val columns = trainData.columns.filter(!_.equals(label))
    columns.foreach(input => println(input))

    val assembler = new VectorAssembler()
      .setInputCols(columns)
      .setOutputCol("features")

    val classifier = new GBTClassifier()
      .setSeed(Random.nextLong())
      .setFeaturesCol("features")
      .setLabelCol(label)
      .setPredictionCol("prediction")

    val pipeline = new Pipeline().setStages(Array(assembler, classifier))

    val eval = new MulticlassClassificationEvaluator()
      .setLabelCol(label)
      .setPredictionCol("prediction")

    val gridSearch = new ParamGridBuilder()
      .addGrid(classifier.impurity, Array("gini", "entropy"))
      .addGrid(classifier.maxIter, Array(5, 10, 20))
      .addGrid(classifier.maxBins, Array(5, 10))
      .addGrid(classifier.maxDepth, Array(10, 20))
      .build()

    val cv = new CrossValidator()
      .setSeed(Random.nextLong())
      .setEstimator(pipeline)
      .setEvaluator(eval)
      .setEstimatorParamMaps(gridSearch)
      .setNumFolds(3)

    val model = cv.fit(trainData).bestModel

    val gbtModel = model.asInstanceOf[PipelineModel].stages.last.asInstanceOf[GBTClassificationModel]

    println(gbtModel.extractParamMap)

    gbtModel.featureImportances.toArray.zip(columns).sorted.reverse.foreach(println)

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

}
