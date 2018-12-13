import org.nd4j.linalg.factory.Nd4j
import utils.IoUEvaluator

object EvaluatorTest extends App{

  val y =     Nd4j.create(Array[Double](1,0,1,1))
  val y_hat = Nd4j.create(Array[Double](1,0,1,0))

  val eval:IoUEvaluator = IoUEvaluator(Seq(y,y,y),Seq(y_hat,y_hat,y_hat))
  println(eval.eval().getOrElse(None))

}
