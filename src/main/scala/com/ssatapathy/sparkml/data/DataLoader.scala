package com.ssatapathy.sparkml.data

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.sql.types.{IntegerType, StringType}
import org.apache.spark.sql.{DataFrame, SparkSession}

class DataLoader(private val spark: SparkSession) {

  import spark.implicits._

  def getCovTypeData: DataFrame = {
    val dataWithoutHeader = spark.read
      .option("inferSchema", true)
      .option("header", false)
      .csv("file:///Users/satapath/work/sparkml/data/covtype.data")

    val colNames = Seq(
      "Elevation", "Aspect", "Slope",
      "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
      "Horizontal_Distance_To_Roadways",
      "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
      "Horizontal_Distance_To_Fire_Points"
    ) ++
      (0 until 4).map(i => s"Wilderness_Area_$i") ++
      (0 until 40).map(i => s"Soil_Type_$i") ++
      Seq("Cover_Type")

    val data = dataWithoutHeader
      .toDF(colNames:_*)
      .withColumn("Cover_Type", $"Cover_Type".cast("double"))

    data
  }

  def getDiabetesData: DataFrame = {
    val dataWithHeader = spark.read
      .option("header", true)
      .option("inferSchema", true)
      .csv("/Users/satapath/work/sparkml/data/diabetes.csv")

    dataWithHeader.toDF()
  }

  def getMushroomsData: DataFrame = {
    val dataWithHeader = spark.read
      .option("header", true)
      .option("inferSchema", true)
      .csv("/Users/satapath/work/sparkml/data/mushrooms.csv")

    val data = dataWithHeader.toDF()
    val stringIndexers = data.columns.map(col => getStringIndexer(col))
    val pipeline = new Pipeline().setStages(stringIndexers)
    val indexedData = pipeline.fit(data).transform(data)
    indexedData.select(indexedData.columns.filter(_.contains("-index")).map(indexedData(_)): _*)
  }

  def getAmesHousingData: DataFrame = {
    val trainDataWithHeader = spark.read
      .option("header", true)
      .option("inferSchema", true)
      .csv("/Users/satapath/work/sparkml/data/ames_housing/train.csv")

    getIndexedAmesHousingData(trainDataWithHeader)
  }

  def getOttoData: DataFrame = {
    val trainDataWithHeader = spark.read
      .option("header", true)
      .option("inferSchema", true)
      .csv("/Users/satapath/work/sparkml/data/otto/train.csv")

    val stringIndexer = getStringIndexer("target")

    val pipeline = new Pipeline().setStages(Array(stringIndexer))

    val indexedData = pipeline.fit(trainDataWithHeader.toDF())
      .transform(trainDataWithHeader.toDF())
      .drop("id", "target")

    indexedData.printSchema()
    indexedData
  }

  def getStringIndexer(col: String): StringIndexer = {
    new StringIndexer()
      .setInputCol(col)
      .setOutputCol(col + "-index")
      .setHandleInvalid("skip")
  }

  private def getIndexedAmesHousingData(dataWithHeader: DataFrame) = {
    val data = dataWithHeader.toDF().drop("Id")
    data.printSchema()

    val stringColumns = data.schema.fields.filter(field =>
      field.dataType.isInstanceOf[StringType]).map(field => field.name)

    val integerColumns = data.schema.fields.filter(field =>
      field.dataType.isInstanceOf[IntegerType]).map(field => field.name)

    val stringIndexers = stringColumns.map(col => getStringIndexer(col))

    val pipeline = new Pipeline().setStages(stringIndexers)
    val stringIndexedData = pipeline.fit(data).transform(data).drop(stringColumns: _*)

    val columns = stringIndexedData.columns.filter(!_.equals("SalePrice"))

    val assembler = new VectorAssembler()
      .setInputCols(columns)
      .setOutputCol("features")

    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(20)

    val indexedData = new Pipeline()
      .setStages(Array(assembler, featureIndexer))
      .fit(stringIndexedData)
      .transform(stringIndexedData)

    indexedData.printSchema()
    indexedData
  }

}
