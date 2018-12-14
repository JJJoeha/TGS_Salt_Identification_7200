package utils

import java.io.File

import org.apache.log4j.{BasicConfigurator, Logger}
import org.deeplearning4j.nn.api.Model
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
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

  def loadModel(modelPath:String, modelType:String): Either[ComputationGraph,MultiLayerNetwork] = {
    log.info("Loading model...")
    modelType.toLowerCase match {
      case "cg" => Left(ModelSerializer.restoreComputationGraph(modelPath))
      case "mn" => Right(ModelSerializer.restoreMultiLayerNetwork(modelPath))
    }
  }

}
