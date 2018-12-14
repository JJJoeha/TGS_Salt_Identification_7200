package Process

import org.apache.log4j.{BasicConfigurator, Logger}
import utils.DataSetLoader

object PreSaveDataSet extends App{

  protected val log : Logger = Logger.getLogger(UNET_RES_TRAIN.getClass)
  BasicConfigurator.configure()

  val mainPath:String = System.getProperty("user.dir")

  val height:Int=101
  val width:Int=101
  val channels:Int=1
  val seed:Long=12345
  val splitRate:Double=0.7
  val batchSize:Int=16
  val epochs:Int=1

  val dataSetPath:String=mainPath+"/Dataset/train"
  val preSavedDataSetPath:String=mainPath+"/Dataset/presaved"

  val dsLoader = DataSetLoader(dataSetPath,height,width,channels,batchSize, seed)
  dsLoader.preSaveDataset(preSavedDataSetPath)

}
