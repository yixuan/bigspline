package statr.stat695ss

import breeze.linalg._
import breeze.numerics._
import scala.util.control._


class BasicSpline(val dat_x: DenseMatrix[Double],
                  val dat_y: DenseVector[Double],
                  val lambda: Double) {
    private val dim_n = dat_x.rows
    private val dim_p = dat_x.cols
    private val dim_m = 1 + dim_p

    private def k1(x: Double) = x - 0.5
    private def k2(x: Double): Double = {
        val k1x = k1(x)
        return 0.5 * (k1x * k1x - 1.0 / 12)
    }
    private def k4(x: Double): Double = {
        val k1x = k1(x)
        val k1xsq = k1x * k1x
        return (k1xsq * k1xsq - k1xsq * 0.5 + 7.0 / 240) / 24
    }
    private def Rkernel(x: Double, y: Double) = k2(x) * k2(y) - k4(math.abs(x - y))
    private def Rkernel(x: DenseVector[Double], y: DenseVector[Double],
                        weights: DenseVector[Double]): Double = {
        val len = x.length
        val res = DenseVector.zeros[Double](len)
        for (i <- 0 until len) {
            res(i) = Rkernel(x(i), y(i))
        }
        return res.dot(weights)
    }
    private def Rkernel(x: DenseVector[Double],
                        y: DenseVector[Double]): Double = {
        val weights = DenseVector.ones[Double](x.length)
        return Rkernel(x, y, weights)
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
    private def mat_prod(x: DenseVector[Double]): DenseVector[Double] = {
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
}
