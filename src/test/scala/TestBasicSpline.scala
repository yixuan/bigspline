package statr.stat695ss

import org.scalatest.FunSuite
import breeze.linalg._
import breeze.numerics._

class TestBasicSpline extends TestBase {

    val f = "other/dat.txt"
    val (x, y) = read_data(f)

    test("Basic Spline") {
        val mod = new BasicSpline(x, y)
        mod.set_opts(-1, 1e-3)
        mod.fit(0.001)
        info("Coefficients = ")
        info(format_vec(mod.coef))
        info("Predicted values = ")
        info(format_vec(mod.pred))
        info("True y values = ")
        info(format_vec(y))
        info("# of iterations = " + mod.niter)
    }
}
