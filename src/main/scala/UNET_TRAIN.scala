import java.util.Random
import java.io.File

import org.datavec.api.split.FileSplit
import org.datavec.api.io.labels.ParentPathLabelGenerator
import org.datavec.api.io.filters.BalancedPathFilter
import org.datavec.image.loader.NativeImageLoader
import org.datavec.image.transform._
import org.datavec.image.recordreader.ImageRecordReader
import org.deeplearning4j.datasets.iterator.file.FileDataSetIterator
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.SplitTestAndTrain
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.dataset.api.preprocessor.{DataNormalization, ImagePreProcessingScaler}
import org.apache.log4j.Logger
import org.apache.log4j.BasicConfigurator
import org.nd4j.linalg.api.ndarray.INDArray;

object UNET_TRAIN extends App {

  protected val log : Logger = Logger.getLogger(UNET_TRAIN.getClass)
  BasicConfigurator.configure()

  val height:Int=101
  val width:Int=101
  val channels:Int=1
  val seed1:Long=12345
  val seed2:Long=42
  val rng1=new Random(seed1)
  val rng2=new Random(seed2)
  val splitRate:Double=0.7
  val batchSize:Int=32
  val epochs:Int=50

  val unet : UNET_RES = UNET_RES(seed1,channels,height,width)

  log.info("Loading data...")

  val mainPath:String = System.getProperty("user.dir")
  val imgPath:File = new File(mainPath, "Dataset/train/images")
  val maskPath:File = new File(mainPath, "Dataset/train/masks")
  val imgNum :Int = Option(imgPath.list).map(_.filter(_.endsWith(".png")).size).getOrElse(0)

  log.info(s"$imgNum images found.")

  val imgSplit = new FileSplit(imgPath, NativeImageLoader.ALLOWED_FORMATS, rng1)
  val maskSplit = new FileSplit(maskPath, NativeImageLoader.ALLOWED_FORMATS, rng1)


//  val recordReader: ImageRecordReader = new ImageRecordReader(height, width, channels)
////  recordReader.initialize(imgSplit)
//
//  val scaler: DataNormalization = new ImagePreProcessingScaler(0, 1)
//
//  val dataIter = new RecordReaderDataSetIterator.Builder(recordReader,batchSize)
//    .preProcessor(scaler)
//    .build

//  dataIter.setCollectMetaData(true)
  print(unet.init.summary())

//  def colloectMask(di:RecordReaderDataSetIterator):RecordReaderDataSetIterator = {
//    di.next
//  }

}
