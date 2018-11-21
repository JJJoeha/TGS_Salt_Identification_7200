
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.conf._
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator
import org.deeplearning4j.nn.transferlearning.TransferLearning
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.Adam
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.nn.conf.layers._
import org.deeplearning4j.nn.conf.graph.L2NormalizeVertex
import org.deeplearning4j.nn.conf.graph.MergeVertex
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.WorkspaceMode

import scala.collection.JavaConversions._
import scala.collection.JavaConverters._
import java.util.Random

case class U_net_Res(batchSize:Int,
                     inputSize:Int,
                     outputSize:Int,
                     inputShape:Array[Int])
                    (implicit
                     val randomSeed:Int=123,
                     val iterations:Int=1) {

  val transferFunction = Activation.RELU


  def graphConf(): ComputationGraphConfiguration = {
    val embeddingSize = 128


    val graph = new NeuralNetConfiguration.Builder()
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .updater(new Adam(0.1, 0.9, 0.999, 0.01)).weightInit(WeightInit.RELU).l2(5e-5)
      .miniBatch(true).convolutionMode(ConvolutionMode.Same)
      .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
      .trainingWorkspaceMode(WorkspaceMode.ENABLED)
      .graphBuilder


    graph
      .addInputs("input")
      .addLayer("test_1", new DenseLayer.Builder().nIn(2).nOut(4).activation(Activation.RELU).build, "input")
      .addLayer("test_2", new DenseLayer.Builder().nIn(4).nOut(8).activation(Activation.RELU).build, "test_1")
      .addLayer("test_3", new DenseLayer.Builder().nIn(8).nOut(8).activation(Activation.RELU).build, "test_2")
      .addLayer("test_4", new DenseLayer.Builder().nIn(8).nOut(4).activation(Activation.RELU).build, "test_3")
      .addLayer("test_5", new DenseLayer.Builder().nIn(4).nOut(2).activation(Activation.RELU).build, "test_4")
      .addLayer("LossLayer", new OutputLayer.Builder(LossFunctions.LossFunction.SQUARED_LOSS).activation(Activation.SOFTMAX).nIn(2).nOut(outputSize).gradientNormalization(GradientNormalization.RenormalizeL2PerLayer).build, "test_5")
      .setOutputs("LossLayer")
      .setInputTypes(InputType.feedForward(4))
      .setBackprop(true)


//      .addLayer("stem-cnn1", new ConvolutionLayer.Builder(Array[Int](7,7), Array[Int](2,2), Array[Int](3,3)).nIn(inputShape(0)).nOut(64).cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE).build, "input1")
//      .addLayer("stem-batch1", new BatchNormalization.Builder(false).nIn(64).nOut(64).build, "stem-cnn1").addLayer("stem-activation1", new ActivationLayer.Builder().activation(Activation.RELU).build, "stem-batch1")
//      .addLayer("stem-pool1", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, Array[Int](3, 3), Array[Int](2, 2), Array[Int](1, 1)).build, "stem-activation1")
//      .addLayer("stem-lrn1", new LocalResponseNormalization.Builder(1, 5, 1e-4, 0.75).build, "stem-pool1")
//      .addLayer("inception-2-cnn1", new ConvolutionLayer.Builder(1,1).nIn(64).nOut(64).cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE).build, "stem-lrn1")
//      .addLayer("inception-2-batch1", new BatchNormalization.Builder(false).nIn(64).nOut(64).build, "inception-2-cnn1")
//      .addLayer("inception-2-activation1", new ActivationLayer.Builder().activation(Activation.RELU).build, "inception-2-batch1")
//      .addLayer("inception-2-cnn2", new ConvolutionLayer.Builder(Array[Int](3, 3), Array[Int](1, 1), Array[Int](1, 1)).nIn(64).nOut(192).cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE).build, "inception-2-activation1").addLayer("inception-2-batch2", new BatchNormalization.Builder(false).nIn(192).nOut(192).build, "inception-2-cnn2")
//      .addLayer("inception-2-activation2", new ActivationLayer.Builder().activation(Activation.RELU).build, "inception-2-batch2")
//      .addLayer("inception-2-lrn1", new LocalResponseNormalization.Builder(1, 5, 1e-4, 0.75).build, "inception-2-activation2")
//      .addLayer("inception-2-pool1", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, Array[Int](3, 3), Array[Int](2, 2), Array[Int](1, 1)).build, "inception-2-lrn1")
//
//    // Inception 3a
//    FaceNetHelper.appendGraph(graph, "3a", 192, Array[Int](3, 5), Array[Int](1, 1), Array[Int](128, 32), Array[Int](96, 16, 32, 64), SubsamplingLayer.PoolingType.MAX, transferFunction, "inception-2-pool1")
//
//    // Inception 3b
//    FaceNetHelper.appendGraph(graph, "3b", 256, Array[Int](3, 5), Array[Int](1, 1), Array[Int](128, 64), Array[Int](96, 32, 64, 64), SubsamplingLayer.PoolingType.PNORM, 2, transferFunction, "inception-3a")
//
//    // Inception 3c
//    graph.addLayer("3c-1x1", new ConvolutionLayer.Builder(Array[Int](1, 1), Array[Int](1, 1)).nIn(320).nOut(128).cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE).build, "inception-3b").addLayer("3c-1x1-norm", FaceNetHelper.batchNorm(128, 128), "3c-1x1").addLayer("3c-transfer1", new ActivationLayer.Builder().activation(transferFunction).build, "3c-1x1-norm").addLayer("3c-3x3", new ConvolutionLayer.Builder(Array[Int](3, 3), Array[Int](2, 2)).nIn(128).nOut(256).cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE).build, "3c-transfer1").addLayer("3c-3x3-norm", FaceNetHelper.batchNorm(256, 256), "3c-3x3").addLayer("3c-transfer2", new ActivationLayer.Builder().activation(transferFunction).build, "3c-3x3-norm").addLayer("3c-2-1x1", new ConvolutionLayer.Builder(Array[Int](1, 1), Array[Int](1, 1)).nIn(320).nOut(32).cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE).build, "inception-3b").addLayer("3c-2-1x1-norm", FaceNetHelper.batchNorm(32, 32), "3c-2-1x1").addLayer("3c-2-transfer3", new ActivationLayer.Builder().activation(transferFunction).build, "3c-2-1x1-norm").addLayer("3c-2-5x5", new ConvolutionLayer.Builder(Array[Int](3, 3), Array[Int](2, 2)).nIn(32).nOut(64).cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE).build, "3c-2-transfer3").addLayer("3c-2-5x5-norm", FaceNetHelper.batchNorm(64, 64), "3c-2-5x5").addLayer("3c-2-transfer4", new ActivationLayer.Builder().activation(transferFunction).build, "3c-2-5x5-norm").addLayer("3c-pool", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, Array[Int](3, 3), Array[Int](2, 2), Array[Int](1, 1)).build, "inception-3b").addVertex("inception-3c", new MergeVertex, "3c-transfer2", "3c-2-transfer4", "3c-pool")
//
//    // Inception 4a
//    FaceNetHelper.appendGraph(graph, "4a", 640, Array[Int](3, 5), Array[Int](1, 1), Array[Int](192, 64), Array[Int](96, 32, 128, 256), SubsamplingLayer.PoolingType.PNORM, 2, transferFunction, "inception-3c")
//
//    // Inception 4e
//    graph
//      .addLayer("4e-1x1", new ConvolutionLayer.Builder(Array[Int](1, 1), Array[Int](1, 1)).nIn(640).nOut(160).cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE).build, "inception-4a")
//      .addLayer("4e-1x1-norm", FaceNetHelper.batchNorm(160, 160), "4e-1x1").addLayer("4e-transfer1", new ActivationLayer.Builder().activation(transferFunction).build, "4e-1x1-norm")
//      .addLayer("4e-3x3", new ConvolutionLayer.Builder(Array[Int](3, 3), Array[Int](2, 2)).nIn(160).nOut(256).cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE).build, "4e-transfer1")
//      .addLayer("4e-3x3-norm", FaceNetHelper.batchNorm(256, 256), "4e-3x3").addLayer("4e-transfer2", new ActivationLayer.Builder().activation(transferFunction).build, "4e-3x3-norm")
//      .addLayer("4e-2-1x1", new ConvolutionLayer.Builder(Array[Int](1, 1), Array[Int](1, 1)).nIn(640).nOut(64).cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE).build, "inception-4a")
//      .addLayer("4e-2-1x1-norm", FaceNetHelper.batchNorm(64, 64), "4e-2-1x1").addLayer("4e-2-transfer3", new ActivationLayer.Builder().activation(transferFunction).build, "4e-2-1x1-norm")
//      .addLayer("4e-2-5x5", new ConvolutionLayer.Builder(Array[Int](3, 3), Array[Int](2, 2)).nIn(64).nOut(128).cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE).build, "4e-2-transfer3")
//      .addLayer("4e-2-5x5-norm", FaceNetHelper.batchNorm(128, 128), "4e-2-5x5").addLayer("4e-2-transfer4", new ActivationLayer.Builder().activation(transferFunction).build, "4e-2-5x5-norm")
//      .addLayer("4e-pool", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, Array[Int](3, 3), Array[Int](2, 2), Array[Int](1, 1)).build, "inception-4a")
//      .addVertex("inception-4e", new MergeVertex, "4e-transfer2", "4e-2-transfer4", "4e-pool")
//
//    // Inception 5a
//    graph
//      .addLayer("5a-1x1", new ConvolutionLayer.Builder(Array[Int](1, 1), Array[Int](1, 1)).nIn(1024).nOut(256).cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE).build, "inception-4e")
//      .addLayer("5a-1x1-norm", FaceNetHelper.batchNorm(256, 256), "5a-1x1")
//      .addLayer("5a-transfer1", new ActivationLayer.Builder().activation(transferFunction).build, "5a-1x1-norm")
//      .addLayer("5a-2-1x1", new ConvolutionLayer.Builder(Array[Int](1, 1), Array[Int](1, 1)).nIn(1024).nOut(96).cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE).build, "inception-4e")
//      .addLayer("5a-2-1x1-norm", FaceNetHelper.batchNorm(96, 96), "5a-2-1x1").addLayer("5a-2-transfer2", new ActivationLayer.Builder().activation(transferFunction).build, "5a-2-1x1-norm")
//      .addLayer("5a-2-3x3", new ConvolutionLayer.Builder(Array[Int](3, 3), Array[Int](1, 1)).nIn(96).nOut(384).cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE).build, "5a-2-transfer2")
//      .addLayer("5a-2-3x3-norm", FaceNetHelper.batchNorm(384, 384), "5a-2-3x3").addLayer("5a-transfer3", new ActivationLayer.Builder().activation(transferFunction).build, "5a-2-3x3-norm").addLayer("5a-3-pool", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.PNORM, Array[Int](3, 3), Array[Int](1, 1)).pnorm(2).build, "inception-4e")
//      .addLayer("5a-3-1x1reduce", new ConvolutionLayer.Builder(Array[Int](1, 1), Array[Int](1, 1)).nIn(1024).nOut(96).cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE).build, "5a-3-pool")
//      .addLayer("5a-3-1x1reduce-norm", FaceNetHelper.batchNorm(96, 96), "5a-3-1x1reduce").addLayer("5a-3-transfer4", new ActivationLayer.Builder().activation(Activation.RELU).build, "5a-3-1x1reduce-norm")
//      .addVertex("inception-5a", new MergeVertex, "5a-transfer1", "5a-transfer3", "5a-3-transfer4")
//
//    // Inception 5b
//    graph
//      .addLayer("5b-1x1", new ConvolutionLayer.Builder(Array[Int](1, 1), Array[Int](1, 1)).nIn(736).nOut(256).cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE).build, "inception-5a")
//      .addLayer("5b-1x1-norm", FaceNetHelper.batchNorm(256, 256), "5b-1x1").addLayer("5b-transfer1", new ActivationLayer.Builder().activation(transferFunction).build, "5b-1x1-norm")
//      .addLayer("5b-2-1x1", new ConvolutionLayer.Builder(Array[Int](1, 1), Array[Int](1, 1)).nIn(736).nOut(96).cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE).build, "inception-5a")
//      .addLayer("5b-2-1x1-norm", FaceNetHelper.batchNorm(96, 96), "5b-2-1x1").addLayer("5b-2-transfer2", new ActivationLayer.Builder().activation(transferFunction).build, "5b-2-1x1-norm")
//      .addLayer("5b-2-3x3", new ConvolutionLayer.Builder(Array[Int](3, 3), Array[Int](1, 1)).nIn(96).nOut(384).cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE).build, "5b-2-transfer2")
//      .addLayer("5b-2-3x3-norm", FaceNetHelper.batchNorm(384, 384), "5b-2-3x3").addLayer("5b-2-transfer3", new ActivationLayer.Builder().activation(transferFunction).build, "5b-2-3x3-norm")
//      .addLayer("5b-3-pool", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, Array[Int](3, 3), Array[Int](1, 1), Array[Int](1, 1)).build, "inception-5a")
//      .addLayer("5b-3-1x1reduce", new ConvolutionLayer.Builder(Array[Int](1, 1), Array[Int](1, 1)).nIn(736).nOut(96).cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE).build, "5b-3-pool")
//      .addLayer("5b-3-1x1reduce-norm", FaceNetHelper.batchNorm(96, 96), "5b-3-1x1reduce").addLayer("5b-3-transfer4", new ActivationLayer.Builder().activation(transferFunction).build, "5b-3-1x1reduce-norm").addVertex("inception-5b", new MergeVertex, "5b-transfer1", "5b-2-transfer3", "5b-3-transfer4")
//
//    // output
//    graph
//      .addLayer("avgpool", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG, Array[Int](3, 3), Array[Int](3, 3)).build, "inception-5b")
//      .addLayer("bottleneck", new DenseLayer.Builder().nIn(736).nOut(embeddingSize).activation(Activation.IDENTITY).build, "avgpool")
//      .addVertex("embeddings", new L2NormalizeVertex(Array[Int](), 1e-6), "bottleneck")
//      .addLayer("lossLayer", new CenterLossOutputLayer.Builder().lossFunction(LossFunctions.LossFunction.SQUARED_LOSS).activation(Activation.SOFTMAX).nIn(128).nOut(outputSize).lambda(1e-4).alpha(0.9).gradientNormalization(GradientNormalization.RenormalizeL2PerLayer).build, "embeddings")
//      .setOutputs("lossLayer")
//      .backprop(true).pretrain(false)
//      .setInputTypes(InputType.convolutional(inputShape(2), inputShape(1), inputShape(0)))

    graph.build
  }

  def summary(): Unit ={
    println(net.summary())
  }

  lazy val net = new ComputationGraph(graphConf())

  net.setListeners(new ScoreIterationListener(1))

}