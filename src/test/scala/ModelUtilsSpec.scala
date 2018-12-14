import org.scalatest.{FlatSpec, Matchers}
import utils.ModelUtils

class ModelUtilsSpec extends FlatSpec with Matchers {
  "ModelUtils" should "be able to save and load models" in {
    val mainPath=System.getProperty("user.dir")
    val model=ModelUtils.loadModel(mainPath+"/src/test/resources/ModelUtilsTest/UNET_RES.zip", "cg")
    ModelUtils.saveModel(model.left.get,mainPath+"/src/test/resources/ModelUtilsTest/UNET_RES_test_save.zip",false)
  }
}
