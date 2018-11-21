import org.apache.spark.sql.{Dataset,Row,SparkSession}
import org.apache.spark.ml.image.ImageSchema

import org.nd4j.linalg.api.ndarray._
import org.nd4j.linalg.factory.Nd4j


object Main extends App{

//  val spark = SparkSession
//    .builder()
//    .appName("Sparktest")
//    .master("local[*]")
//    .getOrCreate()
//
//  val images_train = ImageSchema.
//    readImages("Dataset/images",spark,
//      false,0,
//      false,1.0,1)


//  val sample_NDARRAY:INDArray = Nd4j.zeros(4,2)
  val input_sample :INDArray = Nd4j.ones(2)
  val label_sample :INDArray = Nd4j.ones(1)


  val netSample = U_net_Res(4,2,1,Array(4,2))

  netSample.summary()
  print(input_sample.toString+"\n")
  print(label_sample.toString)
  netSample.net.fit(Array(input_sample), Array(label_sample))


}
