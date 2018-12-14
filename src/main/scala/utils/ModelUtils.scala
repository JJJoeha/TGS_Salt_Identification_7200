package utils

import java.io.File
import org.apache.log4j.{BasicConfigurator, Logger}
import org.deeplearning4j.nn.api.Model
import org.deeplearning4j.util.ModelSerializer

object ModelUtils {

  protected val log : Logger = Logger.getLogger(ModelUtils.getClass)
  BasicConfigurator.configure()

  def saveModel(model:Model, modelSavePath:String, saveUpdater:Boolean) ={
    log.info(s"Saving model to $modelSavePath...")
    try {
      val saveTo: File = new File(modelSavePath)
      saveTo.createNewFile()
      ModelSerializer.writeModel(model, saveTo, saveUpdater)
      log.info("Model saved!")
    }catch{
      case e:Throwable => log.info("Saving failed", e)
    }
  }

  def loadModel(modelPath:String, modelType:String): Model = {
    log.info("Loading model...")
    modelType.toLowerCase match {
      case "cg" => ModelSerializer.restoreComputationGraph(modelPath);
      case "mn" => ModelSerializer.restoreMultiLayerNetwork(modelPath)
    }
  }

//  def calculateIoU(model:Model, testImgPath:String):Option[Double] = {
//      val dataImport:DataImport = DataImport(testImgPath,)
//  }

}
