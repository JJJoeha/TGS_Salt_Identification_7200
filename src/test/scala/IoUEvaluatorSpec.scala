import org.nd4j.linalg.factory.Nd4j
import org.scalatest.{FlatSpec, _}
import utils.IoUEvaluator

class IoUEvaluatorSpec extends FlatSpec with Matchers {
  behavior of "IoUEvaluator"

  it should "correctly calculate average IoU of two sequences of INDArrays" in {

    val y = Nd4j.create(Array[Double](1, 0, 1, 1))
    val y_hat = Nd4j.create(Array[Double](1, 0, 1, 0))

    val eval: IoUEvaluator = IoUEvaluator(Seq(y, y, y), Seq(y_hat, y_hat, y_hat))
    assertResult(eval.eval())(Some(0.75))
  }

  it should "skip INDArrays with different shapes" in {
    val y = Nd4j.create(Array[Double](1, 0, 1, 1))
    val y_hat = Nd4j.create(Array[Double](1, 0, 1, 1))
    val y_troublemaker = Nd4j.create(Array[Double](1))

    val eval: IoUEvaluator = IoUEvaluator(Seq(y, y, y_troublemaker), Seq(y_hat, y_hat, y_hat))
    assertResult(eval.eval())(Some(1))
  }

}
