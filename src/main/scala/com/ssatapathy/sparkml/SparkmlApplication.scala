package com.ssatapathy.sparkml

import com.ssatapathy.sparkml.data.DataLoader
import com.ssatapathy.sparkml.ml.{AnomalyDetector, DecisionForest, GradientBoostedTrees, LR}
import com.typesafe.scalalogging.slf4j.LazyLogging
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.springframework.boot.SpringApplication
import org.springframework.boot.autoconfigure.SpringBootApplication
import org.springframework.boot.context.properties.EnableConfigurationProperties

@SpringBootApplication
@EnableConfigurationProperties
class SparkmlApplication extends LazyLogging {

}

object SparkmlApplication {

  def main(args: Array[String]) {
    val sources: Array[AnyRef] = Array(classOf[SparkmlApplication])
    val context = SpringApplication.run(sources, args)

    val spark = SparkSession.builder()
      .master("local[*]")
      .getOrCreate()

    runLR(spark)

    System.exit(SpringApplication.exit(context))
  }

  private def runAnomalyDetection(spark: SparkSession) = {
    try {
      val anomalyDetector = new AnomalyDetector(spark)
      val data: DataFrame = anomalyDetector.getData
      anomalyDetector.run(data)
    } catch {
      case e: Exception => e.printStackTrace()
    }
  }

  private def runDecisionForest(spark: SparkSession) = {
    try {
      val dataLoader = new DataLoader(spark)
      val decisionForest = new DecisionForest(spark)
      val data = dataLoader.getAmesHousingData

      println(data.head)
      data.show(10, truncate = false)

      val Array(trainData, testData) = data.randomSplit(Array(0.9, 0.1))
      trainData.cache()
      testData.cache()

      decisionForest.buildAndValidateRandomForestRegressor(trainData, testData, "SalePrice")

      trainData.unpersist()
      testData.unpersist()
    } catch {
      case e: Exception => e.printStackTrace()
    }
  }

  private def runGBT(spark: SparkSession) = {
    try {
      val dataLoader = new DataLoader(spark)
      val gbt = new GradientBoostedTrees(spark)
      val data = dataLoader.getAmesHousingData

      val Array(trainData, testData) = data.randomSplit(Array(0.9, 0.1))
      trainData.cache()
      testData.cache()

      gbt.buildAndValidateGBTRegressor(trainData, testData, "SalePrice")

      trainData.unpersist()
      testData.unpersist()
    } catch {
      case e: Exception => e.printStackTrace()
    }
  }

  private def runLR(spark: SparkSession): Unit = {
    try {
      val dataLoader = new DataLoader(spark)
      val lr = new LR(spark)
      val data = dataLoader.getOttoData

      data.show(10, truncate = false)

      val Array(trainData, testData) = data.randomSplit(Array(0.9, 0.1))
      trainData.cache()
      testData.cache()

      lr.buildAndValidateMulticlassClassifier(trainData, testData, "target-index")

      trainData.unpersist()
      testData.unpersist()
    } catch {
      case e: Exception => e.printStackTrace()
    }
  }

}
