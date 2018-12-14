import java.io.File
import java.util.Random

import org.deeplearning4j.util.ModelSerializer
import org.apache.log4j.{BasicConfigurator, Logger}
import org.datavec.api.split.FileSplit
import org.datavec.image.loader.NativeImageLoader
import org.datavec.image.recordreader.ImageRecordReader
import org.datavec.image.transform.{ImageTransform, ResizeImageTransform}
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.nd4j.linalg.dataset.api.preprocessor.{DataNormalization, ImagePreProcessingScaler}
import org.nd4j.linalg.indexing.conditions.GreaterThan
import utils.IoUEvaluator

object EvaluatingProcess extends App{

  protected val log : Logger = Logger.getLogger(EvaluatingProcess.getClass)
  BasicConfigurator.configure()

  val mainPath:String = System.getProperty("user.dir")

  val height:Int=101
  val width:Int=101
  val channels:Int=1
  val seed:Long=12345
  val splitRate:Double=0.7
  val batchSize:Int=32
  val modelSavePath:String=mainPath+"/Saved_Models/UNET.zip"

  val imgPath:File = new File(mainPath, "Dataset/train/images")
  val maskPath:File = new File(mainPath, "Dataset/train/masks")
  val imgNum :Int = Option(imgPath.list).map(_.filter(_.endsWith(".png")).size).getOrElse(0)
  val maskNum:Int = Option(maskPath.list).map(_.filter(_.endsWith(".png")).size).getOrElse(0)

  if(imgNum != maskNum) throw new RuntimeException("Number of images and masks do not match! ")

  log.info(s"$imgNum images and masks found.")

  val imgSplit = new FileSplit(imgPath, NativeImageLoader.ALLOWED_FORMATS, new Random(seed))
  val maskSplit = new FileSplit(maskPath, NativeImageLoader.ALLOWED_FORMATS, new Random(seed))

  val imgReader: ImageRecordReader = new ImageRecordReader(height, width, channels)
  val maskReader: ImageRecordReader = new ImageRecordReader(height,width, channels)

  val resize: ImageTransform = new ResizeImageTransform(width, height)

  imgReader.initialize(imgSplit, resize)
  maskReader.initialize(maskSplit)


  val scaler: DataNormalization = new ImagePreProcessingScaler(0,1)

  val imgIter = new RecordReaderDataSetIterator.Builder(imgReader, batchSize).preProcessor(scaler).build
  val maskIter = new RecordReaderDataSetIterator.Builder(maskReader, batchSize).build

  val net = ModelSerializer.restoreComputationGraph(modelSavePath)
//
//  val cond1=new GreaterThan(0.5)
//
////  val out=net.output(imgIter.next(200).getFeatures).map(_.cond(cond1))
//  val hat=maskIter.next(200).asList()
//  println("SampleOut: "+out)
//  println("SampleLabel: "+hat)
//
//  log.info("Evaluating...")
//
//  val eval:IoUEvaluator = IoUEvaluator(out, hat)
//  println("IoU: "+ eval.eval())
}
