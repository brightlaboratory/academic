/* SimpleApp.scala */

import org.apache.spark.SparkContext
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.{LinearRegression, RandomForestRegressor}
import org.apache.spark.sql.functions.{udf, _}
import org.apache.spark.sql.{DataFrame, SparkSession}

object SimpleApp {

  def countWords(sc: SparkContext): Unit = {
    val pathToFiles = "/Users/newpc/work/research/healthsage/pom.xml"
    val files = sc.textFile(pathToFiles)
    println("Count: " + files.count())
  }

  def createDf(spark: SparkSession) = {
    //Reading Inpatient_prospective_Payment_2015
    val df1 = spark.read
      .option("header", "true") //reading the headers
      .csv(getClass.getClassLoader.getResource("train.csv").getPath)

//    import df1.sparkSession.implicits._

    val df2 = spark.read
      .option("header", "true") //reading the headers
      .csv(getClass.getClassLoader.getResource("test.csv").getPath)

    val df = df1.withColumn("istest", lit(0)).union(df2.withColumn("istest", lit(1)))

    df.printSchema()
    applyLinearRegression(df
      .withColumn("salary_num", toDouble(df("salary")))
    )
  }

  def createDf2(spark: SparkSession) = {
    //Reading Inpatient_prospective_Payment_2015
    val df1 = spark.read
      .option("header", "true") //reading the headers
      .csv(getClass.getClassLoader.getResource("solution.csv").getPath)

    //    import df1.sparkSession.implicits._

//    df1.printSchema()
    processRatings(df1)
  }

  def processRatings(df: DataFrame) = {
    val df2 = df.drop("Age?").drop("Gender").drop("Preferred Genres")
    df2.columns.foreach(v => println("COLUMN: " + v))

    val df3 = df2.columns.toBuffer.foldLeft(df2)((current, c) =>current
      .withColumn(c, col(c).cast("double")))

    df3.take(10).foreach(v => println("ROW: " + v))
    df3.printSchema()
    df3.describe().show()
  }

  val toDouble = udf((str: String) => {
    if (str != null) {
      str.toDouble
    } else {
      0
    }
  })

  def stringIndex(df: DataFrame, inCol: String, outCol: String) = {
    val indexer = new StringIndexer().setInputCol(inCol).setOutputCol(outCol)
      .setHandleInvalid("keep")
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
    println("df7.count(): " + df7.count())

    val trainingData = df7.where(df("istest").equalTo(0))
    val testData = df7.where(df("istest").equalTo(1))

//    val lr = new LinearRegression()
    val lr = new RandomForestRegressor()
      .setMaxBins(10000)


    // Fit the model
    val lrModel = lr.setLabelCol("salary_num").fit(trainingData)
//    println("lrModel: " + lrModel.toDebugString)
    println("lrModel.featureImportances: " + lrModel.featureImportances)

    // Apply the model on testData
    val predictions_lr = lrModel.transform(testData).cache()
    predictions_lr.show(5, truncate = false)
    predictions_lr.printSchema()

    predictions_lr.select("id", "prediction").withColumnRenamed("prediction", "salary")
      .coalesce(1).write.option("header", "true").csv("sampleSubmissionRF3.csv")
  }
}