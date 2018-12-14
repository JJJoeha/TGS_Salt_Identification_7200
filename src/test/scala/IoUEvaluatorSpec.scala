import org.nd4j.linalg.factory.Nd4j
import org.scalatest.{FlatSpec, _}
import utils.IoUEvaluator

class IoUEvaluatorSpec extends FlatSpec with Matchers {
  behavior of "IoUEvaluator"

  it should "correctly calculate average IoU of two sequences of INDArrays" in {

    val y1 = Nd4j.create(Array[Double](1, 0, 1, 1))
    val y2 = Nd4j.create(Array[Double](1, 0, 1, 0))

    val eval: IoUEvaluator = IoUEvaluator(Seq(y1, y1, y2 ,y1), Seq(y2, y2, y2,y1))
    assertResult(eval.eval())(Some(0.875))
  }

  it should "skip INDArrays with different shapes" in {
    val y = Nd4j.create(Array[Double](1, 0, 0, 1))
    val y_hat = Nd4j.create(Array[Double](1, 0, 1, 0))
    val y_troublemaker = Nd4j.create(Array[Double](1))

    val eval: IoUEvaluator = IoUEvaluator(Seq(y, y, y_troublemaker), Seq(y_hat, y_hat, y_hat))
    assertResult(eval.eval())(Some(0.5))
  }

}
