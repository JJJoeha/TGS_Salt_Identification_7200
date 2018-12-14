package Process

import java.io.File
import java.util.Calendar
import java.util.concurrent.TimeUnit

import model_structure.{UNET, UNET_RES}
import org.apache.log4j.{BasicConfigurator, Logger}
import org.deeplearning4j.earlystopping.{EarlyStoppingConfiguration, EarlyStoppingModelSaver}
import org.deeplearning4j.earlystopping.saver.{LocalFileGraphSaver, LocalFileModelSaver}
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator
import org.deeplearning4j.earlystopping.termination.{MaxEpochsTerminationCondition, MaxTimeIterationTerminationCondition}
import org.deeplearning4j.earlystopping.trainer.{EarlyStoppingGraphTrainer, EarlyStoppingTrainer}
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.stats.StatsListener
import org.deeplearning4j.ui.storage.{FileStatsStorage, InMemoryStatsStorage}
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import utils.{DataSetLoader, ModelUtils}

object Training extends App {

  protected val log : Logger = Logger.getLogger(Training.getClass)
  BasicConfigurator.configure()

  val mainPath:String = System.getProperty("user.dir")

  val height:Int=96
  val width:Int=96
  val channels:Int=1
  val seed:Long=12345
  val splitRate:Double=0.7
  val batchSize:Int=16
  val epochs:Int=5

  val network : UNET = UNET(seed,channels,height,width)

  val dataSetPath:String=mainPath+"/Dataset/train"
  val modelSavePath:String=mainPath+s"/saved_model/UNET.zip"
  val preSavedDataSetPath:String=mainPath+"/Dataset/presaved"
  val testImgPath:String=mainPath+"/src/test/resources/TestDataSet/presaved"

  val dsLoader = DataSetLoader(dataSetPath,height,width,channels,batchSize,splitRate, seed)
  val dsi = dsLoader.getTrainIter

  val net = network.init_model
  log.info("Printing network summary...")
  println(net.summary)


  val uiServer = UIServer.getInstance
  val statsStorage = new InMemoryStatsStorage
  uiServer.attach(statsStorage)
  net.setListeners(new StatsListener(statsStorage, 5))


  val esconf:EarlyStoppingConfiguration[ComputationGraph] = new EarlyStoppingConfiguration.Builder[ComputationGraph]
    .epochTerminationConditions(new MaxEpochsTerminationCondition(epochs))
    .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(20, TimeUnit.MINUTES))
    .scoreCalculator(new DataSetLossCalculator(dsLoader.getTestIter, true))
    .evaluateEveryNEpochs(1)
    .modelSaver(new LocalFileGraphSaver(modelSavePath))
    .build

  val trainer  = new EarlyStoppingGraphTrainer(esconf, net, dsi)


  log.info(s"Start training...")
  for(i<-1 to epochs){
    log.info(s"Epoch $i, Time:${Calendar.getInstance.getTime}")
    trainer.fit()
  }


}
