package Process

import org.apache.log4j.{BasicConfigurator, Logger}
import org.deeplearning4j.nn.graph.ComputationGraph
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.indexing.conditions.{Condition,GreaterThan}
import utils.{DataSetLoader, IoUEvaluator, ModelUtils}

object Evaluating extends App{

  protected val log : Logger = Logger.getLogger(Evaluating.getClass)
  BasicConfigurator.configure()

  val mainPath:String = System.getProperty("user.dir")

  val height:Int=96
  val width:Int=96
  val channels:Int=1
  val seed:Long=12345
  val splitRate:Double=0.7
  val batchSize:Int=16

  val modelSavePath:String=mainPath+"/saved_model/bestGraph.bin"
  val net : ComputationGraph = ModelUtils.loadModel(modelSavePath,"cg").left.get

  val dataSetPath:String=mainPath+"/Dataset/train"

  val dsLoader = DataSetLoader(dataSetPath,height,width,channels,batchSize, 0.9, seed)
  log.info("Fetching test data...")
  val tds = dsLoader.getTestData()

  val cond :Condition = new GreaterThan(0.5)

  log.info("Fetching labels...")
  val ys:Seq[INDArray] = tds.asList.toArray(new Array[DataSet](tds.asList.size)).toSeq.map(_.getLabels)

  log.info("Calculating outputs...")
  val input_array:Array[INDArray] = tds.asList.toArray(new Array[DataSet](tds.asList.size)).map(_.getFeatures.div(255))
  val y_hats:Seq[INDArray] = input_array.map(x=>net.outputSingle(x).cond(cond))

  val IouEval:IoUEvaluator = IoUEvaluator(ys,y_hats)
  log.info("Evaluating...")

  IouEval.eval() match {
    case Some(v) => log.info(s"Evaluated IoU: $v")
    case None => log.info("Got no evaluation!")
  }

}
