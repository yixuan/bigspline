package statr.stat695ss

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.apache.spark.broadcast.Broadcast
import org.apache.log4j.Logger
import org.apache.log4j.Level
import breeze.linalg._
import breeze.numerics._
import scala.util.control._
import scala.util.Random

// Store some static methods
object BigSpline {
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
        val len = x.size
        return x.data.zip(y.data).zip(weights.data).map(t => Rkernel(t._1._1, t._1._2) * t._2).sum
    }

    // Generate lambda sequence
    def generate_lambda(lmax: Double = 1.0, lmin: Double = 1e-5, nlambda: Int = 10): Array[Double] = {
        val logmin = math.log(lmin)
        val logmax = math.log(lmax)
        val step = (logmax - logmin) / (nlambda - 1.0)
        val lambda = (0 until nlambda).map(i => math.exp(logmax - i * step)).toArray
        return lambda
    }

    // Nicely print vectors
    private def format_vec(v: DenseVector[Double]): String = {
        return "vec[" + v.data.map(x => "%.5f".format(x)).mkString(", ") + "]"
    }

    // Main program
    // spark-submit --class statr.stat695ss.BigSpline --master local[4] bigspline_2.10-1.0.jar
    def main(args: Array[String]) {
        val conf = new SparkConf().setAppName("Big Spline")
        val sc = new SparkContext(conf)

        Logger.getLogger("org").setLevel(Level.WARN);
        Logger.getLogger("akka").setLevel(Level.WARN);

        val f = "file:///home/qyx/dat.txt"
        val txt = sc.textFile(f, 5)

        val daty = txt.map(line => line.split(' ')(0).toDouble)
        val datx = txt.map(line => DenseVector(line.split(' ').drop(1).map(_.toDouble)))

        daty.cache()
        datx.cache()

        println("\nBuilding model...")
        val model = new BigSpline(datx, daty, sc, true)

        model.set_opts(-1, 1e-3)

        val tune_res = model.tune(1e-6, 1e-2)
        println("\nLambdas = ")
        println(format_vec(tune_res._1))
        println("\nV scores = ")
        println(format_vec(tune_res._2))
        val lambda = tune_res._1(argmin(tune_res._2))
        println("\nSelected lambda = " + lambda)

        model.fit(lambda)
        println("\nCoefficients = ")
        println(format_vec(model.coef))
        println("\nPredicted values = ")
        println(format_vec(model.pred))
        println("\n# of iterations = " + model.niter)
    }
}

class BigSpline(val dat_x: RDD[DenseVector[Double]],
                val dat_y: RDD[Double],
                @transient val sc: SparkContext,
                val cache_Q: Boolean = false) extends CGUpdater with Serializable {
    // Trigger static block
    BigSpline

    // Collected data
    private val xc = dat_x.collect
    private val yc = DenseVector(dat_y.collect)
    private val broad_x = sc.broadcast(xc)

    // Dimensions
    private val dim_n = xc.size
    private val dim_p = xc(0).size
    private val dim_m = 1 + dim_p

    // Estimate weights for the kernels
    private def estimate_theta(): DenseVector[Double] = {
        // Rs(i, j) = Rkernel(x_ij, x_ij)
        val Rs = xc.map(xi => xi.map(x => BigSpline.Rkernel(x, x)))
        // Calculate column sums
        val colsum = Rs.reduce(_ + _)
        return 1.0 / colsum
    }

    // Estimated theta
    private val theta = estimate_theta()

    // Penalty parameter
    private var lambda = 0.0

    // Use the estimated theta to calculate kernel
    private def Rkernel(x: DenseVector[Double], y: DenseVector[Double]): Double = {
        return BasicSpline.Rkernel(x, y, theta)
    }

    // S = (1, x1, x2, ..., x_p)
    private val S = new DenseMatrix(dim_m, dim_n, xc.flatMap(x => Array(1.0) ++ x.toArray)).t

    // Q_ij = R(x_i, x_j), x_i is the i-th row of x
    private val Q = dat_x.map(xi => DenseVector(broad_x.value.map(xj => Rkernel(xi, xj))))
    if(cache_Q)  Q.cache()

    // Q * v
    private def mat_prod(A: RDD[DenseVector[Double]], x: DenseVector[Double]): DenseVector[Double] = {
        return DenseVector(A.map(a => a.dot(x)).collect)
    }

    // T = (S, Q)
    // Ty = T' * y
    private val Tty = DenseVector.zeros[Double](dim_m + dim_n)
    Tty(0 until dim_m) := S.t * yc
    Tty(dim_m to -1) := mat_prod(Q, yc)

    // Matrix-vector product
    // Calculate (T'T + n * lambda * Qs) * x
    // Qs = [O  O]
    //    = [O  Q]
    def mat_prod(x: DenseVector[Double]): DenseVector[Double] = {
        val x1 = x(0 until dim_m)
        val x2 = x(dim_m to -1)
        // Q * x2
        val Qx2 = mat_prod(Q, x2)
        // T * x
        val Tx = (S * x1) + Qx2
        // T' * T * x
        val res = DenseVector.zeros[Double](dim_m + dim_n)
        res(0 until dim_m) := S.t * Tx
        res(dim_m to -1) := mat_prod(Q, Tx)
        // T' * T * x + n * lambda * Qs * x
        res(dim_m to -1) :+= (dim_n * lambda * Qx2)
        // Add a small value to diagonal to make the matrix positive definite
        res :+= (1e-6 * x)
        return res
    }

    private val solver = new ConjugateGradient(this, dim_m + dim_n)
    private var maxit = -1
    private var eps = 1e-6
    private var logs = true

    def set_opts(maxit: Int = -1, eps: Double = 1e-6, logs: Boolean = true) {
        this.maxit = maxit
        this.eps = eps
        this.logs = logs
        solver.set_opts(maxit, eps)
    }

    private def Vscore(lambda: Double, w: DenseVector[Double], Ttw: DenseVector[Double],
                       init: DenseVector[Double]): (Double, DenseVector[Double]) = {
        this.lambda = lambda
        solver.solve(Ttw)
        val w_hat = this.pred

        solver.solve(Tty, init)
        val resid = yc - this.pred

        val denom = 1.0 - w.dot(w_hat) / dim_n.toDouble
        val vscore = resid.dot(resid) / dim_n.toDouble / denom / denom
        return (vscore, solver.coef)
    }

    def tune(lmax: Double, lmin: Double, nlambda: Int = 10, seed: Int = 1): (DenseVector[Double], DenseVector[Double]) = {
        val lambdas = BigSpline.generate_lambda(lmax, lmin, nlambda)
        val vscore = DenseVector.zeros[Double](nlambda)

        val r = new Random(seed)
        val w = new DenseVector((0 until dim_n).map(x => r.nextGaussian()).toArray)
        val Ttw = DenseVector.zeros[Double](dim_m + dim_n)
        Ttw(0 until dim_m) := S.t * w
        Ttw(dim_m to -1) := mat_prod(Q, w)

        // Cache the result from previous lambda
        val last_sol = DenseVector.zeros[Double](dim_m + dim_n)

        if(logs)  println("\nTuning lambdas...")
        for (i <- 0 until nlambda) {
            if(logs)  println("\n===> lambda = " + lambdas(i))
            val res = Vscore(lambdas(i), w, Ttw, last_sol)
            vscore(i) = res._1
            last_sol := res._2
            if(logs)  println("===> v score = " + vscore(i))
        }

        return (DenseVector(lambdas), vscore)
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
        return S * d + mat_prod(Q, c)
    }
}
