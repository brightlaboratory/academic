/* SimpleApp.scala */

import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.{DataFrame, SparkSession}

object SimpleApp {

  def countWords(sc: SparkContext): Unit = {
    val pathToFiles = "/Users/newpc/work/research/healthsage/pom.xml"
    val files = sc.textFile(pathToFiles)
    println("Count: " + files.count())
  }

  def createDf(spark: SparkSession) = {
    //Reading Inpatient_prospective_Payment_2015
    val df = spark.read
      .option("header", "true") //reading the headers
      .csv(getClass.getClassLoader.getResource("train.csv").getPath)

    df.printSchema()
    applyLinearRegression(df
      .withColumn("salary_num", toDouble(df("salary")))
    )
  }

  val toDouble = udf((str: String) => {
    str.toDouble
  })

  def stringIndex(df: DataFrame, inCol: String, outCol: String) = {
    val indexer = new StringIndexer().setInputCol(inCol).setOutputCol(outCol)
      .setHandleInvalid("skip")
    indexer.fit(df).transform(df)
  }

  def applyLinearRegression(df: DataFrame) = {
    val df1 = stringIndex(df, "department_code", "department_code_num")
    val df3 = stringIndex(df1, "worker_group_code", "worker_group_code_num")
    val df4 = stringIndex(df3, "union_code", "union_code_num")
    val df5 = stringIndex(df4, "job_group_code", "job_group_code_num")
    val df6 = stringIndex(df5, "job_code", "job_code_num")

    val assembler = new VectorAssembler().setInputCols(Array("worker_group_code_num",
      "department_code_num",
      "union_code_num", "job_group_code_num",
      "job_code_num")).setOutputCol("features")

    val df7 = assembler.transform(df6)

    val splitSeed = 5043
    val Array(trainingDataOrig, testDataOrig) = df7.randomSplit(Array(0.7, 0.3), splitSeed)

    val trainingData = trainingDataOrig.cache()
    val testData = testDataOrig.cache()
    import trainingData.sparkSession.implicits._

    val lr = new LinearRegression()

    // Fit the model
    val lrModel = lr.setLabelCol("salary_num").fit(trainingData)
    println("lrModel: " + lrModel)

    // Apply the model on testData
    val predictions_lr = lrModel.transform(testData).cache()
    predictions_lr.show(5, truncate = false)
    predictions_lr.printSchema()
  }
}