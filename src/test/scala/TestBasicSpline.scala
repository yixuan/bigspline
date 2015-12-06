package statr.stat695ss

import org.scalatest.FunSuite
import breeze.linalg._
import breeze.numerics._

class TestBasicSpline extends TestBase {

    val n = 20
    val p = 3
    val x = DenseMatrix.rand(n, p)
    val y = DenseVector.rand(n)

    test("Basic Spline") {
        val mod = new BasicSpline(x, y, 0.001)
        mod.fit()
        info(format_vec(mod.coef))
    }
}
