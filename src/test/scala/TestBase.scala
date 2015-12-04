package statr.stat695ss

import org.scalatest.FunSuite
import breeze.linalg._
import breeze.numerics._

abstract class TestBase extends FunSuite {

    def format_vec(v: DenseVector[Double]): String = {
        return "vec[" + v.data.map(x => "%.3f".format(x)).mkString(", ") + "]"
    }
}
