package Process

import java.io.File
import java.util.{Calendar}

import model_structure.{UNET, UNET_RES}
import org.apache.log4j.{BasicConfigurator, Logger}
import org.deeplearning4j.datasets.iterator.{AsyncDataSetIterator}
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.stats.StatsListener
import org.deeplearning4j.ui.storage.{FileStatsStorage, InMemoryStatsStorage}
import org.nd4j.linalg.dataset.{DataSet, ExistingMiniBatchDataSetIterator}
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import utils.{DataSetLoader, ModelUtils}

object UNET_RES_TRAIN extends App {

  protected val log : Logger = Logger.getLogger(UNET_RES_TRAIN.getClass)
  BasicConfigurator.configure()

  val mainPath:String = System.getProperty("user.dir")


  val height:Int=101
  val width:Int=101
  val channels:Int=1
  val seed:Long=12345
  val splitRate:Double=0.7
  val batchSize:Int=16
  val epochs:Int=8

  val network : UNET = UNET(seed,channels,height,width)

  val modelSavePath:String=mainPath+s"/saved_model/${network.getClass.getName}.zip"
  val dataSetPath:String=mainPath+"Dataset/train"
  val preSavedDataSetPath:String=mainPath+"/Dataset/presaved"

  val edsi:DataSetIterator = new ExistingMiniBatchDataSetIterator(new File(preSavedDataSetPath), "ds-%d.bin")
  val dsi = new AsyncDataSetIterator(edsi)

  val net = network.init_model
  log.info("Printing network summary...")
  println(net.summary)


  val uiServer = UIServer.getInstance
  val statsStorage = new InMemoryStatsStorage //Alternative: new FileStatsStorage(File), for saving and loading later
  uiServer.attach(statsStorage)
  net.setListeners(new StatsListener(statsStorage))


  log.info(s"Start training...")
  for(i<-1 to epochs){
    log.info(s"Epoch $i, Time:${Calendar.getInstance.getTime}")
    net.fit(dsi)
  }


  ModelUtils.saveModel(net, modelSavePath, true)

}
