package statr.stat695ss

import org.scalatest.FunSuite
import breeze.linalg._
import breeze.numerics._

class TestLogistic extends TestBase {

    val n = 20
    val mat = DenseMatrix.rand(n, n)
    val A = mat.t * mat
    val b = DenseVector.rand(n)

    val sol = A \ b

    test("Conjugate Gradient") {
        info("True value = ")
        info(format_vec(sol))

        val solver = new SimpleCG(A, b)
        solver.solve(n + 10)
        info("CG value = ")
        info(format_vec(solver.coef))
        info("# of iterations = " + solver.niter)
    }
}
