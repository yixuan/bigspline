package statr.stat695ss

import breeze.linalg._
import breeze.numerics._
import scala.util.control._


class SimpleUpdater(val mat_A: DenseMatrix[Double]) extends CGUpdater {
    // Matrix-vector product
    def mat_prod(x: DenseVector[Double]): DenseVector[Double] = {
        return mat_A * x
    }
}
