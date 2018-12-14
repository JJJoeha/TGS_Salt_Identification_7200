package Process

import java.util.Calendar
import java.util.concurrent.TimeUnit

import org.apache.log4j.{BasicConfigurator, Logger}
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration
import org.deeplearning4j.earlystopping.saver.LocalFileGraphSaver
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator
import org.deeplearning4j.earlystopping.termination.{MaxEpochsTerminationCondition, MaxTimeIterationTerminationCondition}
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingGraphTrainer
import org.deeplearning4j.nn.api.Model
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.stats.StatsListener
import org.deeplearning4j.ui.storage.InMemoryStatsStorage
import utils.{DataSetLoader, ModelUtils}

object Retrain extends App{

  protected val log : Logger = Logger.getLogger(Retrain.getClass)
  BasicConfigurator.configure()

  val mainPath:String = System.getProperty("user.dir")

  val height:Int=96
  val width:Int=96
  val channels:Int=1
  val seed:Long=12345
  val splitRate:Double=0.7
  val batchSize:Int=16
  val epochs:Int=5

  val modelSavePath:String=mainPath+"/saved_model/UNET.zip"
  val net : ComputationGraph = ModelUtils.loadModel(modelSavePath,"cg").left.get

  val dataSetPath:String=mainPath+"/Dataset/train"

  val dsLoader = DataSetLoader(dataSetPath,height,width,channels,batchSize,splitRate, seed)
  val dsi = dsLoader.getTrainIter(1)

  val uiServer = UIServer.getInstance
  val statsStorage = new InMemoryStatsStorage
  uiServer.attach(statsStorage)
  net.setListeners(new StatsListener(statsStorage, 5))

  val esconf:EarlyStoppingConfiguration[ComputationGraph] = new EarlyStoppingConfiguration.Builder[ComputationGraph]
    .epochTerminationConditions(new MaxEpochsTerminationCondition(epochs))
    .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(20, TimeUnit.MINUTES))
    .scoreCalculator(new DataSetLossCalculator(dsLoader.getTestIter(1), true))
    .evaluateEveryNEpochs(1)
    .modelSaver(new LocalFileGraphSaver(modelSavePath))
    .build

  val trainer  = new EarlyStoppingGraphTrainer(esconf,net, dsi)


  log.info(s"Start training...")
  for(i<-1 to epochs){
    log.info(s"Epoch $i, Time:${Calendar.getInstance.getTime}")
    trainer.fit()
  }

}
