package statr.stat695ss

import breeze.linalg._
import breeze.numerics._
import scala.util.control._


class SimpleCG(val mat_A: DenseMatrix[Double], vec_b: DenseVector[Double])
    extends ConjugateGradient(vec_b) {

    // Matrix-vector product to be implemented
    protected def mat_prod(x: DenseVector[Double]): DenseVector[Double] = {
        return mat_A * x
    }
}
