package statr.stat695ss

import breeze.linalg._
import breeze.numerics._
import scala.util.control._


abstract class ConjugateGradient(val n: Integer, val vec_b: DenseVector[Double]) {
    // Matrix-vector product to be implemented
    protected def mat_prod(x: DenseVector[Double]): DenseVector[Double]

    private def square(x: Double): Double = x * x

    val vec_x: DenseVector[Double] = DenseVector.zeros[Double](n)
    val vec_r: DenseVector[Double] = vec_b.copy
    val vec_p: DenseVector[Double] = vec_r.copy

    var rsquare: Double = vec_r.dot(vec_r)
    var iter: Integer = 0

    def solve(maxit: Int = -1, eps: Double = 1e-6) {
        var max_iter: Int = if (maxit < 0) n else maxit

        val loop = new Breaks
        loop.breakable {
            for(i <- 0 until max_iter) {
                iter = i + 1

                val Ap = mat_prod(vec_p)
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

    def coef = vec_x.copy
    def niter = iter
}
