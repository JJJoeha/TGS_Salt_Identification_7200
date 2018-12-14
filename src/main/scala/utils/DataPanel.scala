//package utils
//
//import java.awt.{Graphics2D, Panel}
//
//import org.nd4j.linalg.api.ndarray.INDArray
//
//case class DataPanel(data: INDArray) extends Panel {
//
//  override def paintComponent(g: Graphics2D) {
//    val dx = g.getClipBounds.width.toFloat  / data.size(0)
//    val dy = g.getClipBounds.height.toFloat / data.size(1)
//    for {
//      x <- 0 until data.size(0)
//      y <- 0 until data.size(1)
//      x1 = (x * dx).toInt
//      y1 = (y * dy).toInt
//      x2 = ((x + 1) * dx).toInt
//      y2 = ((y + 1) * dy).toInt
//    } {
//      data(x)(y) match {
//        case 1 => g.setColor(c)
//        case 0 => g.setColor(Color.WHITE)
//      }
//      g.fillRect(x1, y1, x2 - x1, y2 - y1)
//    }
//  }
//}
