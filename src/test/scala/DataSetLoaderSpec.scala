import org.scalatest.{FlatSpec, Matchers}
import utils.{DataSetLoader, ModelUtils}

class DataSetLoaderSpec extends FlatSpec with Matchers {
  "DataSetLoader" should "properly import images" in {
    val mainPath=System.getProperty("user.dir")
    val dsl=DataSetLoader(mainPath+"/src/test/resources/TestDataSet",101,101,1,1,0.7,12345)
    val img=dsl.getImgIter().next().getFeatures.div(255)
    val mask=dsl.getMaskIter().next().getFeatures.div(65535)
    assert(img.equalShapes(mask))
  }
}