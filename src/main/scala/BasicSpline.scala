package statr.stat695ss

import breeze.linalg._
import breeze.numerics._
import scala.util.control._

object BasicSpline {
    // Bernoulli polynomials
    def k1(x: Double) = x - 0.5
    def k2(x: Double): Double = {
        val k1x = k1(x)
        return 0.5 * (k1x * k1x - 1.0 / 12)
    }
    def k4(x: Double): Double = {
        val k1x = k1(x)
        val k1xsq = k1x * k1x
        return (k1xsq * k1xsq - k1xsq * 0.5 + 7.0 / 240) / 24
    }

    // Kernels
    def Rkernel(x: Double, y: Double) = k2(x) * k2(y) - k4(math.abs(x - y))
    def Rkernel(x: DenseVector[Double], y: DenseVector[Double], weights: DenseVector[Double]): Double = {
        val len = x.length
        val res = DenseVector.zeros[Double](len)
        for (i <- 0 until len) {
            res(i) = Rkernel(x(i), y(i))
        }
        return res.dot(weights)
    }
}

class BasicSpline(val dat_x: DenseMatrix[Double],
                  val dat_y: DenseVector[Double]) extends CGUpdater {
    // Trigger static block
    BasicSpline

    // Dimensions
    private val dim_n = dat_x.rows
    private val dim_p = dat_x.cols
    private val dim_m = 1 + dim_p

    // Estimate weights for the kernels
    private def estimate_theta(): DenseVector[Double] = {
        val theta = DenseVector.zeros[Double](dim_p)
        for (i <- 0 until dim_p) {
            theta(i) = 1.0 / sum(dat_x(::, i).map(x => BasicSpline.Rkernel(x, x)))
        }
        return theta
    }

    // Estimated theta
    val theta = estimate_theta()

    // Penalty parameter
    var lambda = 0.0

    // Use the estimated theta to calculate kernel
    private def Rkernel(x: DenseVector[Double], y: DenseVector[Double]): Double = {
        return BasicSpline.Rkernel(x, y, theta)
    }

    // S = (1, x1, x2, ..., x_p)
    private val S = DenseMatrix.ones[Double](dim_n, dim_m)
    S(::, 1 to dim_p) := dat_x

    // Q_ij = R(x_i, x_j), x_i is the i-th row of x
    private val Q = DenseMatrix.zeros[Double](dim_n, dim_n)
    for (i <- 0 until dim_n) {
        for (j <- 0 until dim_n) {
            Q(i, j) = Rkernel(dat_x(i, ::).t, dat_x(j, ::).t)
        }
    }

    // T = (S, Q)
    // Ty = T' * y
    private val Tty = DenseVector.zeros[Double](dim_m + dim_n)
    Tty(0 until dim_m) := S.t * dat_y
    Tty(dim_m to -1) := Q * dat_y

    // Matrix-vector product
    // Calculate (T'T + n * lambda * Qs) * x
    // Qs = [O  O]
    //    = [O  Q]
    def mat_prod(x: DenseVector[Double]): DenseVector[Double] = {
        val x1 = x(0 until dim_m)
        val x2 = x(dim_m to -1)
        // Q * x2
        val Qx2 = Q * x2
        // T * x
        val Tx = (S * x1) + Qx2
        // T' * T * x
        val res = DenseVector.zeros[Double](dim_m + dim_n)
        res(0 until dim_m) := S.t * Tx
        res(dim_m to -1) := Q * Tx
        // T' * T * x + n * lambda * Q * x
        res(dim_m to -1) :+= (dim_n * lambda * Qx2)
        return res
    }

    val solver = new ConjugateGradient(this, dim_m + dim_n)
    var maxit = -1
    var eps = 1e-6

    def set_opts(maxit: Int = -1, eps: Double = 1e-6) {
        this.maxit = maxit
        this.eps = eps
        solver.set_opts(maxit, eps)
    }

    def fit(lambda: Double) {
        this.lambda = lambda
        solver.solve(Tty)
    }

    def coef = solver.coef
    def niter = solver.niter
    def pred: DenseVector[Double] = {
        val cd = solver.coef
        val d = cd(0 until dim_m)
        val c = cd(dim_m to -1)
        return S * d + Q * c
    }
}
