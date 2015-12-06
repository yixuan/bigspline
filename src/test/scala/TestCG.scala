package statr.stat695ss

import org.scalatest.FunSuite
import breeze.linalg._
import breeze.numerics._

class TestCG extends TestBase {

    val n = 20
    val mat = DenseMatrix.rand(n, n)
    val A = mat.t * mat
    val b = DenseVector.rand(n)

    val sol = A \ b

    test("Conjugate Gradient") {
        info("True value = ")
        info(format_vec(sol))

        val updater = new SimpleUpdater(A)
        val solver = new ConjugateGradient(updater, n)
        solver.set_opts(n + 10)
        solver.solve(b)
        info("CG value = ")
        info(format_vec(solver.coef))
        info("# of iterations = " + solver.niter)
    }
}
