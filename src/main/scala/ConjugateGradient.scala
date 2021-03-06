package statr.stat695ss

import breeze.linalg._
import breeze.numerics._
import scala.util.control._

// A provider should implement the x => A * x operation
trait CGUpdater {
    def mat_prod(x: DenseVector[Double]): DenseVector[Double]
}

// Solving linear equation A * x = b using conjugate gradient method
class ConjugateGradient(val updater: CGUpdater, val ncoef: Int) extends Serializable {

    private def square(x: Double): Double = x * x

    private val vec_x: DenseVector[Double] = DenseVector.zeros[Double](ncoef)
    private var maxit = 0
    private var eps = 0.0
    private var iter = 0

    def set_opts(maxit: Int = -1, eps: Double = 1e-6) {
        this.maxit = if (maxit < 0) ncoef else maxit
        this.eps = eps
    }

    // Solving the linear system with a provided initial guess of x
    def solve(vec_b: DenseVector[Double], init_x: DenseVector[Double]) {
        vec_x := init_x
        val vec_r: DenseVector[Double] = vec_b - updater.mat_prod(vec_x)
        val vec_p: DenseVector[Double] = vec_r.copy

        var rsquare: Double = vec_r.dot(vec_r)
        iter = 0

        val loop = new Breaks
        loop.breakable {
            for(i <- 0 until maxit) {
                iter = i + 1

                val Ap = updater.mat_prod(vec_p)
                val alpha = rsquare / (vec_p.dot(Ap))
                vec_x :+= alpha * vec_p
                vec_r :-= alpha * Ap
                val rnorm = norm(vec_r)

                if (rnorm < eps)
                    loop.break

                val beta = rnorm * rnorm / rsquare
                vec_p :*= beta
                vec_p :+= vec_r

                rsquare = rnorm * rnorm
            }
        }
    }

    // Using zeros as the initial guess
    def solve(vec_b: DenseVector[Double]) {
        solve(vec_b, DenseVector.zeros[Double](ncoef))
    }

    // Solved x
    def coef = vec_x.copy
    // Number of iterations
    def niter = iter
}
