package utils

import java.io.File
import java.util.Random

import org.apache.log4j.{BasicConfigurator, Logger}
import org.datavec.api.split.FileSplit
import org.datavec.image.loader.NativeImageLoader
import org.datavec.image.recordreader.ImageRecordReader
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.datasets.iterator.{ExistingDataSetIterator, SamplingDataSetIterator}
import org.nd4j.linalg.dataset.{DataSet, ExistingMiniBatchDataSetIterator}
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.dataset.api.preprocessor.{DataNormalization, ImagePreProcessingScaler}

case class DataSetLoader(dataSetPath:String,
                         height:Int,
                         width:Int,
                         channels:Int,
                         batchSize:Int,
                         seed:Long) {

  val log : Logger = Logger.getLogger(this.getClass)
  BasicConfigurator.configure()
  val imgPath: File = new File(dataSetPath, "images")
  val maskPath: File = new File(dataSetPath, "masks")
  val imgNum: Int = Option(imgPath.list).map(_.count(_.endsWith(".png"))).getOrElse(0)
  val maskNum: Int = Option(maskPath.list).map(_.count(_.endsWith(".png"))).getOrElse(0)

  if (imgNum != maskNum) throw new RuntimeException("Number of images and masks do not match! ")

  log.info(s"$imgNum images and masks found.")

  val imgSplit = new FileSplit(imgPath, NativeImageLoader.ALLOWED_FORMATS, new Random(seed))
  val maskSplit = new FileSplit(maskPath, NativeImageLoader.ALLOWED_FORMATS, new Random(seed))

  val imgReader: ImageRecordReader = new ImageRecordReader(height, width, channels)
  val maskReader: ImageRecordReader = new ImageRecordReader(height, width, channels)

  imgReader.initialize(imgSplit)
  maskReader.initialize(maskSplit)
  val scaler: DataNormalization = new ImagePreProcessingScaler(0, 1)
  val imgIter:DataSetIterator = new RecordReaderDataSetIterator.Builder(imgReader,batchSize).preProcessor(scaler).build
  val maskIter:DataSetIterator = new RecordReaderDataSetIterator.Builder(maskReader,batchSize).build

  lazy val dataset:DataSet = new DataSet(imgIter.next(imgNum).getFeatures, maskIter.next(maskNum).getFeatures.div(65535))

  def load():DataSet = {
    log.info("Loading dataset...")
    dataset
  }

  def getImgIter():DataSetIterator = imgIter
  def getMaskIter():DataSetIterator = maskIter
  def preSaveDataset(saveTo:String)={
    log.info("Saving dataset...")
    val di:DataSetIterator = new SamplingDataSetIterator(dataset, batchSize, imgNum/2)
    var i=0
    while(di.hasNext){
      di.next().save(new File(saveTo,"ds-"+i+".bin"))
      i+=1
    }
    log.info(s"Dataset saved to $saveTo...")
  }


}