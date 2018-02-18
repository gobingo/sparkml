package com.ssatapathy.sparkml.ml

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{DecisionTreeClassifier, RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.feature.{VectorAssembler, VectorIndexer}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._

import scala.util.Random

class DecisionForest(private val spark: SparkSession) {

  import spark.implicits._

  def buildAndValidateSimpleDecisionTree(trainData: DataFrame, testData: DataFrame): Unit = {
    val inputCols = trainData.columns.filter(_ != "Cover_Type")

    val assembler = new VectorAssembler()
      .setInputCols(inputCols)
      .setOutputCol("featureVector")

    val assembledTrainData = assembler.transform(trainData)

    assembledTrainData.select("featureVector").show(truncate = false)

    val classifier = new DecisionTreeClassifier()
      .setSeed(Random.nextLong())
      .setLabelCol("Cover_Type")
      .setFeaturesCol("featureVector")
      .setPredictionCol("prediction")

    val model = classifier.fit(assembledTrainData)

    println(model.toDebugString)

    model.featureImportances.toArray.zip(inputCols)
      .sorted.reverse.foreach(println)

    val assembledTestData = assembler.transform(testData)
    val predictions = model.transform(assembledTestData)

    predictions.select("Cover_Type", "prediction", "probability").show(truncate = false)

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("Cover_Type")
      .setPredictionCol("prediction")

    val precision = evaluator.setMetricName("weightedPrecision").evaluate(predictions)
    val recall = evaluator.setMetricName("weightedRecall").evaluate(predictions)
    val accuracy = evaluator.setMetricName("accuracy").evaluate(predictions)
    val f1_score = evaluator.setMetricName("f1").evaluate(predictions)

    val predictionRDD = predictions
      .select("prediction", "Cover_Type")
      .as[(Double, Double)]
      .rdd

    val multiclassMetrics = new MulticlassMetrics(predictionRDD)

    println("confusion matrix:")
    println(multiclassMetrics.confusionMatrix)

    println("accuracy = " + accuracy +
      ", precision = " + precision +
      ", recall = " + recall +
      ", f1-score = " + f1_score)

    val confusionMatrix = predictions
      .groupBy("Cover_Type")
      .pivot("prediction", 1 to 7)
      .count()
      .na.fill(0.0)
      .orderBy("Cover_Type")

    confusionMatrix.show()
  }

  def randomClassifier(trainData: DataFrame, testData: DataFrame): Unit = {
    val trainPriorProbabilities = classProbabilities(trainData)
    val testPriorProbabilities = classProbabilities(testData)
    val accuracy = trainPriorProbabilities.zip(testPriorProbabilities).map {
      case (trainProb, cvProb) => trainProb * cvProb
    }.sum

    println(accuracy)
  }

  def classProbabilities(data: DataFrame): Array[Double] = {
    val total = data.count()
    data.groupBy("Cover_Type")
      .count()
      .orderBy("Cover_Type")
      .select("count").as[Double]
      .map(_ / total).
      collect()
  }

  def buildAndValidateDecisionTree(trainData: DataFrame, testData: DataFrame): Unit = {
    val inputCols = trainData.columns.filter(_ != "Cover_Type")

    val assembler = new VectorAssembler()
      .setInputCols(inputCols)
      .setOutputCol("featureVector")

    val classifier = new DecisionTreeClassifier()
      .setSeed(Random.nextLong())
      .setLabelCol("Cover_Type")
      .setFeaturesCol("featureVector")
      .setPredictionCol("prediction")

    val pipeline = new Pipeline().setStages(Array(assembler, classifier))

    val paramGrid = new ParamGridBuilder()
      .addGrid(classifier.impurity, Seq("gini", "entropy"))
      .addGrid(classifier.maxDepth, Seq(1, 20))
      .addGrid(classifier.maxBins, Seq(40, 300))
      .addGrid(classifier.minInfoGain, Seq(0.0, 0.05))
      .build()

    val multiclassEval = new MulticlassClassificationEvaluator()
      .setLabelCol("Cover_Type")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val validator = new TrainValidationSplit()
      .setSeed(Random.nextLong())
      .setTrainRatio(0.9)
      .setEstimator(pipeline)
      .setEstimatorParamMaps(paramGrid)
      .setEvaluator(multiclassEval)

    val validatorModel = validator.fit(trainData)

    val paramsAndMetrics = validatorModel.validationMetrics
      .zip(validatorModel.getEstimatorParamMaps)
      .sortBy(-_._1)

    paramsAndMetrics.foreach { case (metric, params) =>
      println(metric)
      println(params)
      println()
    }

    val bestModel = validatorModel.bestModel

    println(bestModel.asInstanceOf[PipelineModel].stages.last.extractParamMap)

    println(validatorModel.validationMetrics.max)

    val testAccuracy = multiclassEval.evaluate(bestModel.transform(testData))
    println("testAccuracy = " + testAccuracy)

    val trainAccuracy = multiclassEval.evaluate(bestModel.transform(trainData))
    println("trainAccuracy = " + trainAccuracy)
  }

  def unencodeOneHot(data: DataFrame): DataFrame = {
    val wildernessCols = (0 until 4).map(i => s"Wilderness_Area_$i").toArray

    val wildernessAssembler = new VectorAssembler()
      .setInputCols(wildernessCols)
      .setOutputCol("wilderness")

    val unhotUDF = udf((vec: Vector) => vec.toArray.indexOf(1.0).toDouble)

    val withWilderness = wildernessAssembler.transform(data)
      .drop(wildernessCols:_*)
      .withColumn("wilderness", unhotUDF($"wilderness"))

    val soilCols = (0 until 40).map(i => s"Soil_Type_$i").toArray

    val soilAssembler = new VectorAssembler()
      .setInputCols(soilCols)
      .setOutputCol("soil")

    soilAssembler.transform(withWilderness)
      .drop(soilCols:_*)
      .withColumn("soil", unhotUDF($"soil"))
  }

  def buildAndValidateDecisionTreeWithCategorical(trainData: DataFrame, testData: DataFrame): Unit = {
    val unencTrainData = unencodeOneHot(trainData)
    val unencTestData = unencodeOneHot(testData)

    val inputCols = unencTrainData.columns.filter(_ != "Cover_Type")

    val assembler = new VectorAssembler()
      .setInputCols(inputCols)
      .setOutputCol("featureVector")

    // Automatically identify categorical features, and index them.
    // Set maxCategories so features with > 40 distinct values are treated as continuous.
    val indexer = new VectorIndexer()
      .setMaxCategories(40)
      .setInputCol("featureVector")
      .setOutputCol("indexedVector")

    val classifier = new DecisionTreeClassifier()
      .setSeed(Random.nextLong())
      .setLabelCol("Cover_Type")
      .setFeaturesCol("indexedVector")
      .setPredictionCol("prediction")

    val pipeline = new Pipeline().setStages(Array(assembler, indexer, classifier))

    val paramGrid = new ParamGridBuilder()
      .addGrid(classifier.impurity, Seq("gini", "entropy"))
      .addGrid(classifier.maxDepth, Seq(1, 20))
      .addGrid(classifier.maxBins, Seq(40, 300))
      .addGrid(classifier.minInfoGain, Seq(0.0, 0.05))
      .build()

    val multiclassEval = new MulticlassClassificationEvaluator()
      .setLabelCol("Cover_Type")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val validator = new TrainValidationSplit()
      .setSeed(Random.nextLong())
      .setEstimator(pipeline)
      .setEvaluator(multiclassEval)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.9)

    val validatorModel = validator.fit(unencTrainData)

    val bestModel = validatorModel.bestModel

    println(bestModel.asInstanceOf[PipelineModel].stages.last.extractParamMap)

    val testAccuracy = multiclassEval.evaluate(bestModel.transform(unencTestData))

    println(testAccuracy)
  }

  def buildAndValidateRandomForestClassifier(trainData: DataFrame, testData: DataFrame, label: String): Unit = {
    val inputCols = trainData.columns.filter(_ != label)

    val assembler = new VectorAssembler()
      .setInputCols(inputCols)
      .setOutputCol("featureVector")

    val classifier = new RandomForestClassifier()
      .setSeed(Random.nextLong())
      .setLabelCol(label)
      .setFeaturesCol("featureVector")
      .setPredictionCol("prediction")
      .setImpurity("entropy")
      .setMaxDepth(20)
      .setMaxBins(20)
      .setNumTrees(100)

    val pipeline = new Pipeline().setStages(Array(assembler, classifier))

    val model = pipeline.fit(trainData)

    val forestModel =
      model.asInstanceOf[PipelineModel].stages.last.asInstanceOf[RandomForestClassificationModel]

    println(forestModel.extractParamMap)

    forestModel.featureImportances.toArray.zip(inputCols).sorted.reverse.foreach(println)

    val multiclassEval = new MulticlassClassificationEvaluator()
      .setLabelCol(label)
      .setPredictionCol("prediction")

    val predictions = model.transform(testData)
    predictions.select(label, "prediction").show()

    val precision = multiclassEval.setMetricName("weightedPrecision").evaluate(predictions)
    val recall = multiclassEval.setMetricName("weightedRecall").evaluate(predictions)
    val accuracy = multiclassEval.setMetricName("accuracy").evaluate(predictions)
    val f1Score = multiclassEval.setMetricName("f1").evaluate(predictions)

    println("accuracy: " + accuracy)
    println("precision: " + precision)
    println("recall: " + recall)
    println("f1Score: " + f1Score)
  }

  def buildAndValidateRandomForestRegressor(trainData: DataFrame, testData: DataFrame, label: String): Unit = {
    val columns = trainData.columns.filter(_ != label)

    val classifier = new RandomForestRegressor()
      .setSeed(Random.nextLong())
      .setFeaturesCol("indexedFeatures")
      .setLabelCol(label)
      .setPredictionCol("prediction")
      .setImpurity("variance")
      .setMaxDepth(20)
      .setMaxBins(328)
      .setNumTrees(100)

    val eval = new RegressionEvaluator()
      .setLabelCol(label)
      .setPredictionCol("prediction")

    val pipeline = new Pipeline().setStages(Array(classifier))

    val model = pipeline.fit(trainData)

    val forestModel = model.asInstanceOf[PipelineModel].stages.last.asInstanceOf[RandomForestRegressionModel]
    println(forestModel.extractParamMap())

    forestModel.featureImportances.toArray.zip(columns).sorted.reverse.foreach(println)

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

}
