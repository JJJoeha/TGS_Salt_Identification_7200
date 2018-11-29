import scala.util.Random
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config._
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf._
import org.deeplearning4j.nn.conf.distribution._
import org.deeplearning4j.nn.conf.{ComputationGraphConfiguration, NeuralNetConfiguration}
import org.deeplearning4j.nn.conf.ConvolutionMode
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.conf.layers._
import org.deeplearning4j.nn.conf.graph.{ElementWiseVertex, MergeVertex}
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.WorkspaceMode
import org.deeplearning4j.nn.weights.WeightInit



case class UNET_RES(seed:Long,
                channels:Int,
                height:Int,
                width:Int){
  import UNET_RES._

  def init: ComputationGraph = {
    val graph : ComputationGraphConfiguration.GraphBuilder = graphBuilder(seed)

    graph.addInputs("input").setInputTypes(InputType.convolutional(height,width,channels))

    val conf = graph.build
    val model = new ComputationGraph(conf)
    model.init()
    model
  }

}


object UNET_RES {

  private val seed:Long=12345
  private val weightInit:WeightInit = WeightInit.RELU
  private val initFilterNum = 16
  private val dropOut = 0.5
  private val updater = new AdaDelta
  private val cacheMode = CacheMode.NONE
  private val workspaceMode = WorkspaceMode.ENABLED
  private val cudnnAlgoMode = ConvolutionLayer.AlgoMode.PREFER_FASTEST

  def graphBuilder(seed:Long) : ComputationGraphConfiguration.GraphBuilder = {

    val builder: ComputationGraphConfiguration.GraphBuilder = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .weightInit(weightInit)
      .dist(new TruncatedNormalDistribution(0.0, 0.5))
      .updater(updater)
      .l2(5e-5)
      .miniBatch(true)
      .cacheMode(cacheMode)
      .trainingWorkspaceMode(workspaceMode)
      .inferenceWorkspaceMode(workspaceMode)
      .graphBuilder()

    var samplingBlockName:String = "sub1"

    //subsampling block 1   101->50
    builder
      .addLayer(samplingBlockName+"_conv", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(initFilterNum)
        .convolutionMode(ConvolutionMode.Same).build, "input")

      //residual block 1
      .addLayer(samplingBlockName+"_res1_BN", new BatchNormalization.Builder()
        .activation(Activation.RELU).build, samplingBlockName+"_conv")
      .addLayer(samplingBlockName+"_res1_conv1", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum)
        .convolutionMode(ConvolutionMode.Same).build, samplingBlockName+"_res1_BN")
      .addLayer(samplingBlockName+"_res1_BN1", new BatchNormalization.Builder()
        .activation(Activation.RELU).build,samplingBlockName+"_res1_conv1")
      .addLayer(samplingBlockName+"_res1_conv2", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum)
        .convolutionMode(ConvolutionMode.Same).build, samplingBlockName+"_res1_BN1")
      .addVertex(samplingBlockName+"_res1_skip", new ElementWiseVertex(ElementWiseVertex.Op.Add),
        samplingBlockName+"_conv", samplingBlockName+"_res1_conv2")

      //residual block 2
      .addLayer(samplingBlockName+"_res2_BN", new BatchNormalization.Builder()
      .activation(Activation.RELU).build, samplingBlockName+"_res1_skip")
      .addLayer(samplingBlockName+"_res2_conv1", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum)
        .convolutionMode(ConvolutionMode.Same).build, samplingBlockName+"_res2_BN")
      .addLayer(samplingBlockName+"_res2_BN1", new BatchNormalization.Builder()
        .activation(Activation.RELU).build,samplingBlockName+"_res2_conv1")
      .addLayer(samplingBlockName+"_res2_conv2", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum)
        .convolutionMode(ConvolutionMode.Same).build, samplingBlockName+"_res2_BN1")
      .addVertex(samplingBlockName+"_res2_skip", new ElementWiseVertex(ElementWiseVertex.Op.Add),
        samplingBlockName+"_res1_skip", samplingBlockName+"_res2_conv2")
      .addLayer(samplingBlockName+"_res2_BN2", new BatchNormalization.Builder()
        .activation(Activation.RELU).build, samplingBlockName+"_res2_skip")

      //pooling
      .addLayer(samplingBlockName+"_pool", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2,2).build,
      samplingBlockName+"_res2_BN2")

      //dropout
      .addLayer(samplingBlockName+"_dropout", new DropoutLayer.Builder(dropOut/2).build, samplingBlockName+"_pool")


    samplingBlockName = "sub2"
    //subsampling block 2   50->25
    builder
      .addLayer(samplingBlockName+"_conv", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(initFilterNum*2)
        .convolutionMode(ConvolutionMode.Same).build, "sub1_dropout")

      //residual block 1
      .addLayer(samplingBlockName+"_res1_BN", new BatchNormalization.Builder()
      .activation(Activation.RELU).build, samplingBlockName+"_conv")
      .addLayer(samplingBlockName+"_res1_conv1", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum*2)
        .convolutionMode(ConvolutionMode.Same).build, samplingBlockName+"_res1_BN")
      .addLayer(samplingBlockName+"_res1_BN1", new BatchNormalization.Builder()
        .activation(Activation.RELU).build,samplingBlockName+"_res1_conv1")
      .addLayer(samplingBlockName+"_res1_conv2", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum*2)
        .convolutionMode(ConvolutionMode.Same).build, samplingBlockName+"_res1_BN1")
      .addVertex(samplingBlockName+"_res1_skip", new ElementWiseVertex(ElementWiseVertex.Op.Add),
        samplingBlockName+"_conv", samplingBlockName+"_res1_conv2")

      //residual block 2
      .addLayer(samplingBlockName+"_res2_BN", new BatchNormalization.Builder()
      .activation(Activation.RELU).build, samplingBlockName+"_res1_conv2")
      .addLayer(samplingBlockName+"_res2_conv1", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum*2)
        .convolutionMode(ConvolutionMode.Same).build, samplingBlockName+"_res2_BN")
      .addLayer(samplingBlockName+"_res2_BN1", new BatchNormalization.Builder()
        .activation(Activation.RELU).build,samplingBlockName+"_res2_conv1")
      .addLayer(samplingBlockName+"_res2_conv2", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum*2)
        .convolutionMode(ConvolutionMode.Same).build, samplingBlockName+"_res2_BN1")
      .addVertex(samplingBlockName+"_res2_skip", new ElementWiseVertex(ElementWiseVertex.Op.Add),
        samplingBlockName+"_res1_skip", samplingBlockName+"_res2_conv2")
      .addLayer(samplingBlockName+"_res2_BN2", new BatchNormalization.Builder()
        .activation(Activation.RELU).build, samplingBlockName+"_res2_skip")

      //pooling
      .addLayer(samplingBlockName+"_pool", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2,2).build,
      samplingBlockName+"_res2_BN2")

      //dropout
      .addLayer(samplingBlockName+"_dropout", new DropoutLayer.Builder(dropOut).build, samplingBlockName+"_pool")


    samplingBlockName = "sub3"
    //subsampling block 3   25->12
    builder
      .addLayer(samplingBlockName+"_conv", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(initFilterNum*4)
        .convolutionMode(ConvolutionMode.Same).build, "sub2_dropout")

      //residual block 1
      .addLayer(samplingBlockName+"_res1_BN", new BatchNormalization.Builder()
      .activation(Activation.RELU).build, samplingBlockName+"_conv")
      .addLayer(samplingBlockName+"_res1_conv1", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum*4)
        .convolutionMode(ConvolutionMode.Same).build, samplingBlockName+"_res1_BN")
      .addLayer(samplingBlockName+"_res1_BN1", new BatchNormalization.Builder()
        .activation(Activation.RELU).build,samplingBlockName+"_res1_conv1")
      .addLayer(samplingBlockName+"_res1_conv2", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum*4)
        .convolutionMode(ConvolutionMode.Same).build, samplingBlockName+"_res1_BN1")
      .addVertex(samplingBlockName+"_res1_skip", new ElementWiseVertex(ElementWiseVertex.Op.Add),
        samplingBlockName+"_conv", samplingBlockName+"_res1_conv2")

      //residual block 2
      .addLayer(samplingBlockName+"_res2_BN", new BatchNormalization.Builder()
      .activation(Activation.RELU).build, samplingBlockName+"_res1_conv2")
      .addLayer(samplingBlockName+"_res2_conv1", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum*4)
        .convolutionMode(ConvolutionMode.Same).build, samplingBlockName+"_res2_BN")
      .addLayer(samplingBlockName+"_res2_BN1", new BatchNormalization.Builder()
        .activation(Activation.RELU).build,samplingBlockName+"_res2_conv1")
      .addLayer(samplingBlockName+"_res2_conv2", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum*4)
        .convolutionMode(ConvolutionMode.Same).build, samplingBlockName+"_res2_BN1")
      .addVertex(samplingBlockName+"_res2_skip", new ElementWiseVertex(ElementWiseVertex.Op.Add),
        samplingBlockName+"_res1_skip", samplingBlockName+"_res2_conv2")
      .addLayer(samplingBlockName+"_res2_BN2", new BatchNormalization.Builder()
        .activation(Activation.RELU).build, samplingBlockName+"_res2_skip")

      //pooling
      .addLayer(samplingBlockName+"_pool", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2,2).build,
      samplingBlockName+"_res2_BN2")

      //dropout
      .addLayer(samplingBlockName+"_dropout", new DropoutLayer.Builder(dropOut).build, samplingBlockName+"_pool")



    samplingBlockName = "sub4"
    //subsampling block 4   12->6
    builder
      .addLayer(samplingBlockName+"_conv", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(initFilterNum*8)
        .convolutionMode(ConvolutionMode.Same).build, "sub3_dropout")

      //residual block 1
      .addLayer(samplingBlockName+"_res1_BN", new BatchNormalization.Builder()
      .activation(Activation.RELU).build, samplingBlockName+"_conv")
      .addLayer(samplingBlockName+"_res1_conv1", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum*8)
        .convolutionMode(ConvolutionMode.Same).build, samplingBlockName+"_res1_BN")
      .addLayer(samplingBlockName+"_res1_BN1", new BatchNormalization.Builder()
        .activation(Activation.RELU).build,samplingBlockName+"_res1_conv1")
      .addLayer(samplingBlockName+"_res1_conv2", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum*8)
        .convolutionMode(ConvolutionMode.Same).build, samplingBlockName+"_res1_BN1")
      .addVertex(samplingBlockName+"_res1_skip", new ElementWiseVertex(ElementWiseVertex.Op.Add),
        samplingBlockName+"_conv", samplingBlockName+"_res1_conv2")

      //residual block 2
      .addLayer(samplingBlockName+"_res2_BN", new BatchNormalization.Builder()
      .activation(Activation.RELU).build, samplingBlockName+"_res1_conv2")
      .addLayer(samplingBlockName+"_res2_conv1", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum*8)
        .convolutionMode(ConvolutionMode.Same).build, samplingBlockName+"_res2_BN")
      .addLayer(samplingBlockName+"_res2_BN1", new BatchNormalization.Builder()
        .activation(Activation.RELU).build,samplingBlockName+"_res2_conv1")
      .addLayer(samplingBlockName+"_res2_conv2", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum*8)
        .convolutionMode(ConvolutionMode.Same).build, samplingBlockName+"_res2_BN1")
      .addVertex(samplingBlockName+"_res2_skip", new ElementWiseVertex(ElementWiseVertex.Op.Add),
        samplingBlockName+"_res1_skip", samplingBlockName+"_res2_conv2")
      .addLayer(samplingBlockName+"_res2_BN2", new BatchNormalization.Builder()
        .activation(Activation.RELU).build, samplingBlockName+"_res2_skip")

      //pooling
      .addLayer(samplingBlockName+"_pool", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2,2).build,
      samplingBlockName+"_res2_BN2")

      //dropout
      .addLayer(samplingBlockName+"_dropout", new DropoutLayer.Builder(dropOut).build, samplingBlockName+"_pool")


    samplingBlockName = "middle"
    //middle block
    builder
      .addLayer(samplingBlockName+"_conv", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(initFilterNum*16)
        .convolutionMode(ConvolutionMode.Same).build, "sub4_dropout")

      //residual block 1
      .addLayer(samplingBlockName+"_res1_BN", new BatchNormalization.Builder()
      .activation(Activation.RELU).build, samplingBlockName+"_conv")
      .addLayer(samplingBlockName+"_res1_conv1", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum*16)
        .convolutionMode(ConvolutionMode.Same).build, samplingBlockName+"_res1_BN")
      .addLayer(samplingBlockName+"_res1_BN1", new BatchNormalization.Builder()
        .activation(Activation.RELU).build,samplingBlockName+"_res1_conv1")
      .addLayer(samplingBlockName+"_res1_conv2", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum*16)
        .convolutionMode(ConvolutionMode.Same).build, samplingBlockName+"_res1_BN1")

      //residual block 2
      .addLayer(samplingBlockName+"_res2_BN", new BatchNormalization.Builder()
      .activation(Activation.RELU).build, samplingBlockName+"_res1_conv2")
      .addLayer(samplingBlockName+"_res2_conv1", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum*16)
        .convolutionMode(ConvolutionMode.Same).build, samplingBlockName+"_res2_BN")
      .addLayer(samplingBlockName+"_res2_BN1", new BatchNormalization.Builder()
        .activation(Activation.RELU).build,samplingBlockName+"_res2_conv1")
      .addLayer(samplingBlockName+"_res2_conv2", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum*16)
        .convolutionMode(ConvolutionMode.Same).build, samplingBlockName+"_res2_BN1")
      .addLayer(samplingBlockName+"_res2_BN2", new BatchNormalization.Builder()
        .activation(Activation.RELU).build, samplingBlockName+"_res2_conv2")



    samplingBlockName = "up4"
    //upSampling block 1, 6->12
    //The upSampling blocks indexes will be reversed to correspond with subSampling blocks
      builder
        //transpose convolution
        .addLayer(samplingBlockName+"_deconv", new Deconvolution2D.Builder(3,3).stride(2,2).nOut(initFilterNum*8)
          .convolutionMode(ConvolutionMode.Same).build, "middle_res2_BN2")
        //Concatenate
        .addVertex(samplingBlockName+"_uconv", new MergeVertex, "up4_deconv", "sub4_res2_BN2")
        //dropout
        .addLayer(samplingBlockName+"_dropout", new DropoutLayer.Builder(dropOut).build, samplingBlockName+"_uconv")
        //conv
        .addLayer(samplingBlockName+"_conv", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(initFilterNum*8)
          .convolutionMode(ConvolutionMode.Same).build, samplingBlockName+"_dropout")

        //residual block 1
        .addLayer(samplingBlockName+"_res1_BN", new BatchNormalization.Builder()
        .activation(Activation.RELU).build, samplingBlockName+"_conv")
        .addLayer(samplingBlockName+"_res1_conv1", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum*8)
          .convolutionMode(ConvolutionMode.Same).build, samplingBlockName+"_res1_BN")
        .addLayer(samplingBlockName+"_res1_BN1", new BatchNormalization.Builder()
          .activation(Activation.RELU).build,samplingBlockName+"_res1_conv1")
        .addLayer(samplingBlockName+"_res1_conv2", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum*8)
          .convolutionMode(ConvolutionMode.Same).build, samplingBlockName+"_res1_BN1")
        .addVertex(samplingBlockName+"_res1_skip", new ElementWiseVertex(ElementWiseVertex.Op.Add),
          samplingBlockName+"_conv", samplingBlockName+"_res1_conv2")

        //residual block 2
        .addLayer(samplingBlockName+"_res2_BN", new BatchNormalization.Builder()
        .activation(Activation.RELU).build, samplingBlockName+"_res1_conv2")
        .addLayer(samplingBlockName+"_res2_conv1", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum*8)
          .convolutionMode(ConvolutionMode.Same).build, samplingBlockName+"_res2_BN")
        .addLayer(samplingBlockName+"_res2_BN1", new BatchNormalization.Builder()
          .activation(Activation.RELU).build,samplingBlockName+"_res2_conv1")
        .addLayer(samplingBlockName+"_res2_conv2", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum*8)
          .convolutionMode(ConvolutionMode.Same).build, samplingBlockName+"_res2_BN1")
        .addVertex(samplingBlockName+"_res2_skip", new ElementWiseVertex(ElementWiseVertex.Op.Add),
          samplingBlockName+"_res1_skip", samplingBlockName+"_res2_conv2")
        .addLayer(samplingBlockName+"_res2_BN2", new BatchNormalization.Builder()
          .activation(Activation.RELU).build, samplingBlockName+"_res2_skip")


    samplingBlockName = "up3"
    //upSampling block 2,  12->25
    builder
      //transpose convolution
      .addLayer(samplingBlockName+"_deconv", new Deconvolution2D.Builder(3,3).stride(2,2).nOut(initFilterNum*4)
      .convolutionMode(ConvolutionMode.Truncate).build, "up4_res2_BN2")
      //Concatenate
      .addVertex(samplingBlockName+"_uconv", new MergeVertex, "up3_deconv", "sub3_res2_BN2")
      //dropout
      .addLayer(samplingBlockName+"_dropout", new DropoutLayer.Builder(dropOut).build, samplingBlockName+"_uconv")
      //conv
      .addLayer(samplingBlockName+"_conv", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(initFilterNum*4)
      .convolutionMode(ConvolutionMode.Same).build, samplingBlockName+"_dropout")

      //residual block 1
      .addLayer(samplingBlockName+"_res1_BN", new BatchNormalization.Builder()
      .activation(Activation.RELU).build, samplingBlockName+"_conv")
      .addLayer(samplingBlockName+"_res1_conv1", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum*4)
        .convolutionMode(ConvolutionMode.Same).build, samplingBlockName+"_res1_BN")
      .addLayer(samplingBlockName+"_res1_BN1", new BatchNormalization.Builder()
        .activation(Activation.RELU).build,samplingBlockName+"_res1_conv1")
      .addLayer(samplingBlockName+"_res1_conv2", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum*4)
        .convolutionMode(ConvolutionMode.Same).build, samplingBlockName+"_res1_BN1")
      .addVertex(samplingBlockName+"_res1_skip", new ElementWiseVertex(ElementWiseVertex.Op.Add),
        samplingBlockName+"_conv", samplingBlockName+"_res1_conv2")

      //residual block 2
      .addLayer(samplingBlockName+"_res2_BN", new BatchNormalization.Builder()
      .activation(Activation.RELU).build, samplingBlockName+"_res1_conv2")
      .addLayer(samplingBlockName+"_res2_conv1", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum*4)
        .convolutionMode(ConvolutionMode.Same).build, samplingBlockName+"_res2_BN")
      .addLayer(samplingBlockName+"_res2_BN1", new BatchNormalization.Builder()
        .activation(Activation.RELU).build,samplingBlockName+"_res2_conv1")
      .addLayer(samplingBlockName+"_res2_conv2", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum*4)
        .convolutionMode(ConvolutionMode.Same).build, samplingBlockName+"_res2_BN1")
      .addVertex(samplingBlockName+"_res2_skip", new ElementWiseVertex(ElementWiseVertex.Op.Add),
        samplingBlockName+"_res1_skip", samplingBlockName+"_res2_conv2")
      .addLayer(samplingBlockName+"_res2_BN2", new BatchNormalization.Builder()
        .activation(Activation.RELU).build, samplingBlockName+"_res2_skip")



    samplingBlockName = "up2"
    //upSampling block 3,  25->50
    builder
      //transpose convolution
      .addLayer(samplingBlockName+"_deconv", new Deconvolution2D.Builder(3,3).stride(2,2).nOut(initFilterNum*2)
      .convolutionMode(ConvolutionMode.Same).build, "up3_res2_BN2")
      //Concatenate
      .addVertex(samplingBlockName+"_uconv", new MergeVertex, "up2_deconv", "sub2_res2_BN2")
      //dropout
      .addLayer(samplingBlockName+"_dropout", new DropoutLayer.Builder(dropOut).build, samplingBlockName+"_uconv")
      //conv
      .addLayer(samplingBlockName+"_conv", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(initFilterNum*2)
      .convolutionMode(ConvolutionMode.Same).build, samplingBlockName+"_dropout")

      //residual block 1
      .addLayer(samplingBlockName+"_res1_BN", new BatchNormalization.Builder()
      .activation(Activation.RELU).build, samplingBlockName+"_conv")
      .addLayer(samplingBlockName+"_res1_conv1", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum*2)
        .convolutionMode(ConvolutionMode.Same).build, samplingBlockName+"_res1_BN")
      .addLayer(samplingBlockName+"_res1_BN1", new BatchNormalization.Builder()
        .activation(Activation.RELU).build,samplingBlockName+"_res1_conv1")
      .addLayer(samplingBlockName+"_res1_conv2", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum*2)
        .convolutionMode(ConvolutionMode.Same).build, samplingBlockName+"_res1_BN1")
      .addVertex(samplingBlockName+"_res1_skip", new ElementWiseVertex(ElementWiseVertex.Op.Add),
        samplingBlockName+"_conv", samplingBlockName+"_res1_conv2")

      //residual block 2
      .addLayer(samplingBlockName+"_res2_BN", new BatchNormalization.Builder()
      .activation(Activation.RELU).build, samplingBlockName+"_res1_conv2")
      .addLayer(samplingBlockName+"_res2_conv1", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum*2)
        .convolutionMode(ConvolutionMode.Same).build, samplingBlockName+"_res2_BN")
      .addLayer(samplingBlockName+"_res2_BN1", new BatchNormalization.Builder()
        .activation(Activation.RELU).build,samplingBlockName+"_res2_conv1")
      .addLayer(samplingBlockName+"_res2_conv2", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum*2)
        .convolutionMode(ConvolutionMode.Same).build, samplingBlockName+"_res2_BN1")
      .addVertex(samplingBlockName+"_res2_skip", new ElementWiseVertex(ElementWiseVertex.Op.Add),
        samplingBlockName+"_res1_skip", samplingBlockName+"_res2_conv2")
      .addLayer(samplingBlockName+"_res2_BN2", new BatchNormalization.Builder()
        .activation(Activation.RELU).build, samplingBlockName+"_res2_skip")


    samplingBlockName = "up1"
    //upSampling block 4, 50->101
    builder
      //transpose convolution
      .addLayer(samplingBlockName+"_deconv", new Deconvolution2D.Builder(3,3).stride(2,2).nOut(initFilterNum)
      .convolutionMode(ConvolutionMode.Truncate).build, "up2_res2_BN2")
      //Concatenate
      .addVertex(samplingBlockName+"_uconv", new MergeVertex, "up1_deconv", "sub1_res2_BN2")
      //dropout
      .addLayer(samplingBlockName+"_dropout", new DropoutLayer.Builder(dropOut).build, samplingBlockName+"_uconv")
      //conv
      .addLayer(samplingBlockName+"_conv", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(initFilterNum)
      .convolutionMode(ConvolutionMode.Same).build, samplingBlockName+"_dropout")

      //residual block 1
      .addLayer(samplingBlockName+"_res1_BN", new BatchNormalization.Builder()
      .activation(Activation.RELU).build, samplingBlockName+"_conv")
      .addLayer(samplingBlockName+"_res1_conv1", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum)
        .convolutionMode(ConvolutionMode.Same).build, samplingBlockName+"_res1_BN")
      .addLayer(samplingBlockName+"_res1_BN1", new BatchNormalization.Builder()
        .activation(Activation.RELU).build,samplingBlockName+"_res1_conv1")
      .addLayer(samplingBlockName+"_res1_conv2", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum)
        .convolutionMode(ConvolutionMode.Same).build, samplingBlockName+"_res1_BN1")
      .addVertex(samplingBlockName+"_res1_skip", new ElementWiseVertex(ElementWiseVertex.Op.Add),
        samplingBlockName+"_conv", samplingBlockName+"_res1_conv2")

      //residual block 2
      .addLayer(samplingBlockName+"_res2_BN", new BatchNormalization.Builder()
      .activation(Activation.RELU).build, samplingBlockName+"_res1_conv2")
      .addLayer(samplingBlockName+"_res2_conv1", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum)
        .convolutionMode(ConvolutionMode.Same).build, samplingBlockName+"_res2_BN")
      .addLayer(samplingBlockName+"_res2_BN1", new BatchNormalization.Builder()
        .activation(Activation.RELU).build,samplingBlockName+"_res2_conv1")
      .addLayer(samplingBlockName+"_res2_conv2", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(initFilterNum)
        .convolutionMode(ConvolutionMode.Same).build, samplingBlockName+"_res2_BN1")
      .addVertex(samplingBlockName+"_res2_skip", new ElementWiseVertex(ElementWiseVertex.Op.Add),
        samplingBlockName+"_res1_skip", samplingBlockName+"_res2_conv2")
      .addLayer(samplingBlockName+"_res2_BN2", new BatchNormalization.Builder()
        .activation(Activation.RELU).build, samplingBlockName+"_res2_skip")


    //output
        .addLayer("final_conv_sigmoid", new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(1)
          .activation(Activation.SIGMOID).build, "up1_res2_BN2")
        .addLayer("outputs", new CnnLossLayer.Builder(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY)
          .activation(Activation.SOFTMAX).build, "final_conv_sigmoid")
        .setOutputs("outputs")


    builder
  }


}
