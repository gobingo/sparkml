package com.ssatapathy.sparkml.ml

import org.apache.spark.ml.{PipelineModel, Pipeline}
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.feature.{OneHotEncoder, VectorAssembler, StringIndexer, StandardScaler}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.{DataFrame, SparkSession}
import scala.util.Random

class AnomalyDetector(private val spark: SparkSession) {

  import spark.implicits._

  def getData: DataFrame = {
    val data = spark.read
      .option("inferSchema", true)
      .option("header", false)
      .csv("file:///tmp/data/kddcup.data_10_percent")
      .toDF(
        "duration", "protocol_type", "service", "flag",
        "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
        "hot", "num_failed_logins", "logged_in", "num_compromised",
        "root_shell", "su_attempted", "num_root", "num_file_creations",
        "num_shells", "num_access_files", "num_outbound_cmds",
        "is_host_login", "is_guest_login", "count", "srv_count",
        "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
        "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
        "dst_host_count", "dst_host_srv_count",
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate", "dst_host_srv_serror_rate",
        "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
        "label")

    data.cache()
    data
  }

  def explore(data: DataFrame): Unit = {
    data.groupBy("label").count().orderBy($"count".desc).show(10)
    data.groupBy("protocol_type").count().orderBy($"count".desc).show(10)
  }

  // Clustering, Take 0
  def clusteringTake0(data: DataFrame): Unit = {
    val numericOnly = data.drop("protocol_type", "service", "flag").cache()

    val assembler = new VectorAssembler()
      .setInputCols(numericOnly.columns.filter(_ != "label"))
      .setOutputCol("featureVector")

    val kmeans = new KMeans()
      .setSeed(Random.nextLong())
      .setPredictionCol("cluster")
      .setFeaturesCol("featureVector")

    val pipeline = new Pipeline().setStages(Array(assembler, kmeans))
    val pipelineModel = pipeline.fit(numericOnly)
    val kmeansModel = pipelineModel.stages.last.asInstanceOf[KMeansModel]

    kmeansModel.clusterCenters.foreach(println)

    val withCluster = pipelineModel.transform(numericOnly)

    withCluster.select("cluster", "label")
      .groupBy("cluster", "label").count()
      .orderBy($"cluster", $"count".desc)
      .show(25)

    numericOnly.unpersist()
  }

  // Clustering, Take 1
  def clusteringTake1(data: DataFrame): Unit = {
    val numericOnly = data.drop("protocol_type", "service", "flag").cache()
    //(20 to 100 by 20).map(k => (k, clusteringScore0(numericOnly, k))).foreach(println)
    (20 to 100 by 20).map(k => (k, clusteringScore1(numericOnly, k))).foreach(println)
    numericOnly.unpersist()
  }

  def clusteringScore0(data: DataFrame, k: Int): Double = {
    val assembler = new VectorAssembler()
      .setInputCols(data.columns.filter(_ != "label"))
      .setOutputCol("featureVector")

    val kmeans = new KMeans()
      .setSeed(Random.nextLong())
      .setK(k)
      .setPredictionCol("cluster")
      .setFeaturesCol("featureVector")

    val pipeline = new Pipeline().setStages(Array(assembler, kmeans))

    val kmeansModel = pipeline.fit(data).stages.last.asInstanceOf[KMeansModel]
    kmeansModel.computeCost(assembler.transform(data)) / data.count()
  }

  def clusteringScore1(data: DataFrame, k: Int): Double = {
    val assembler = new VectorAssembler()
      .setInputCols(data.columns.filter(_ != "label"))
      .setOutputCol("featureVector")

    val kmeans = new KMeans()
      .setSeed(Random.nextLong())
      .setK(k)
      .setPredictionCol("cluster")
      .setFeaturesCol("featureVector")
      .setMaxIter(40)
      .setTol(1.0e-5)

    val pipeline = new Pipeline().setStages(Array(assembler, kmeans))

    val kmeansModel = pipeline.fit(data).stages.last.asInstanceOf[KMeansModel]
    kmeansModel.computeCost(assembler.transform(data)) / data.count()
  }

  // Clustering, Take 2
  def clusteringTake2(data: DataFrame): Unit = {
    val numericOnly = data.drop("protocol_type", "service", "flag").cache()
    (60 to 270 by 30).map(k => (k, clusteringScore2(numericOnly, k))).foreach(println)
    numericOnly.unpersist()
  }

  def clusteringScore2(data: DataFrame, k: Int): Double = {
    val assembler = new VectorAssembler()
      .setInputCols(data.columns.filter(_ != "label"))
      .setOutputCol("featureVector")

    val scaler = new StandardScaler()
      .setInputCol("featureVector")
      .setOutputCol("scaledFeatureVector")
      .setWithStd(true)
      .setWithMean(false)

    val kmeans = new KMeans()
      .setSeed(Random.nextLong())
      .setK(k)
      .setPredictionCol("cluster")
      .setFeaturesCol("scaledFeatureVector")
      .setMaxIter(40)
      .setTol(1.0e-5)

    val pipeline = new Pipeline().setStages(Array(assembler, scaler, kmeans))
    val pipelineModel = pipeline.fit(data)

    val kmeansModel = pipelineModel.stages.last.asInstanceOf[KMeansModel]
    kmeansModel.computeCost(pipelineModel.transform(data)) / data.count()
  }

  // Clustering, Take 3
  def clusteringTake3(data: DataFrame): Unit = {
    (20 to 100 by 20).map(k => (k, clusteringScore3(data, k))).foreach(println)
  }

  def clusteringScore3(data: DataFrame, k: Int): Double = {
    val (protoTypeEncoder, protoTypeVecCol) = oneHotPipeline("protocol_type")
    val (serviceEncoder, serviceVecCol) = oneHotPipeline("service")
    val (flagEncoder, flagVecCol) = oneHotPipeline("flag")

    // Original columns, without label / string columns, but with new vector encoded cols
    val assembleCols = Set(data.columns: _*) --
    Seq("label", "protocol_type", "service", "flag") ++
    Seq(protoTypeVecCol, serviceVecCol, flagVecCol)

    val assembler = new VectorAssembler().
    setInputCols(assembleCols.toArray).
    setOutputCol("featureVector")

    val scaler = new StandardScaler()
      .setInputCol("featureVector")
      .setOutputCol("scaledFeatureVector")
      .setWithStd(true)
      .setWithMean(false)

    val kmeans = new KMeans().
    setSeed(Random.nextLong()).
    setK(k).
    setPredictionCol("cluster").
    setFeaturesCol("scaledFeatureVector").
    setMaxIter(40).
    setTol(1.0e-5)

    val pipeline = new Pipeline().setStages(
      Array(protoTypeEncoder, serviceEncoder, flagEncoder, assembler, scaler, kmeans))

    val pipelineModel = pipeline.fit(data)

    val kmeansModel = pipelineModel.stages.last.asInstanceOf[KMeansModel]
    kmeansModel.computeCost(pipelineModel.transform(data)) / data.count()
  }

  def oneHotPipeline(inputCol: String): (Pipeline, String) = {
    val indexer = new StringIndexer().
      setInputCol(inputCol).
      setOutputCol(inputCol + "_indexed")

    val encoder = new OneHotEncoder().
      setInputCol(inputCol + "_indexed").
      setOutputCol(inputCol + "_vec")

    val pipeline = new Pipeline().setStages(Array(indexer, encoder))

    (pipeline, inputCol + "_vec")
  }

  // Clustering, Take 4
  def clusteringTake4(data: DataFrame): Unit = {
    (60 to 270 by 30).map(k => (k, clusteringScore4(data, k))).foreach(println)

    val pipelineModel = fitPipeline4(data, 150)

    val countByClusterLabel = pipelineModel.transform(data).
      select("cluster", "label").
      groupBy("cluster", "label").count().
      orderBy("cluster", "label")

    countByClusterLabel.show()
  }

  def fitPipeline4(data: DataFrame, k: Int): PipelineModel = {
    val (protoTypeEncoder, protoTypeVecCol) = oneHotPipeline("protocol_type")
    val (serviceEncoder, serviceVecCol) = oneHotPipeline("service")
    val (flagEncoder, flagVecCol) = oneHotPipeline("flag")

    // Original columns, without label / string columns, but with new vector encoded cols
    val assembleCols = Set(data.columns: _*) --
      Seq("label", "protocol_type", "service", "flag") ++
      Seq(protoTypeVecCol, serviceVecCol, flagVecCol)

    val assembler = new VectorAssembler().
      setInputCols(assembleCols.toArray).
      setOutputCol("featureVector")

    val scaler = new StandardScaler()
      .setInputCol("featureVector")
      .setOutputCol("scaledFeatureVector")
      .setWithStd(true)
      .setWithMean(false)

    val kmeans = new KMeans().
      setSeed(Random.nextLong()).
      setK(k).
      setPredictionCol("cluster").
      setFeaturesCol("scaledFeatureVector").
      setMaxIter(40).
      setTol(1.0e-5)

    val pipeline = new Pipeline().setStages(
      Array(protoTypeEncoder, serviceEncoder, flagEncoder, assembler, scaler, kmeans))

    pipeline.fit(data)
  }

  def clusteringScore4(data: DataFrame, k: Int): Double = {
    val pipelineModel = fitPipeline4(data, k)

    // Predict cluster for each datum
    val clusterLabel = pipelineModel.transform(data)
      .select("cluster", "label").as[(Int, String)]

    // Extract collections of labels, per cluster
    // Count labels in collections
    val weightedClusterEntropy = clusterLabel
      .groupByKey { case (cluster, _) => cluster }
      .mapGroups { case (_, clusterLabels) =>
        val labels = clusterLabels.map { case (_, label) => label }.toSeq
        val labelCounts = labels.groupBy(identity).values.map(_.size)
        labels.size * entropy(labelCounts)
      }.collect()

    // Average entropy weighted by cluster size
    weightedClusterEntropy.sum / data.count()
  }

  def entropy(counts: Iterable[Int]): Double = {
    val values = counts.filter(_ > 0)
    val n = values.map(_.toDouble).sum
    values.map { v =>
      val p = v / n
      -p * math.log(p)
    }.sum
  }

  // Detect anomalies
  def run(data: DataFrame): Unit = {
    val pipelineModel = fitPipeline4(data, 150)

    val kMeansModel = pipelineModel.stages.last.asInstanceOf[KMeansModel]
    val centroids = kMeansModel.clusterCenters

    centroids.foreach(centroid => println(centroid.toArray.mkString(",")))

    val clustered = pipelineModel.transform(data)
    clustered.select("cluster", "scaledFeatureVector").as[(Int, Vector)].take(100)
      .foreach(e => {
        println(centroids(e._1).toArray.mkString(","))
        println(e._2.toArray.mkString(","))
        println(Vectors.sqdist(centroids(e._1), e._2))
      })

    /*val threshold = clustered.select("cluster", "scaledFeatureVector").as[(Int, Vector)]
      .map { case (cluster, vec) => Vectors.sqdist(centroids(cluster), vec) }
      .orderBy($"value".desc)
      .take(100)
      .last

    println("threshold: " + threshold)

    val originalCols = data.columns

    val anomalies = clustered.filter { row =>
      val cluster = row.getAs[Int]("cluster")
      val vec = row.getAs[Vector]("scaledFeatureVector")
      Vectors.sqdist(centroids(cluster), vec) >= threshold
    }.select(originalCols.head, originalCols.tail: _*)

    println(anomalies.first())*/
  }

}
