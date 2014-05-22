package md

import org.jblas.{SimpleBlas, Eigen, DoubleMatrix}
import scala.util.Random
import scala.collection.JavaConverters._
import cc.factorie.la._
import cc.factorie.variable.DiscreteDomain
import scala.collection.mutable.ArrayBuffer
import cc.factorie.optimize.Example

case class DenseDesignMatrix(mat: BlasDenseTensor2, transposed: Boolean = false) {
  def domainSize = if (transposed) mat.dim1 else mat.dim2
}

object DenseDesignMatrix {
  def fromRowArrays(rows: Seq[Array[Double]]): DenseDesignMatrix =
    DenseDesignMatrix(new BlasDenseTensor2(rows.toArray.flatten, rows.size, rows(0).size), transposed = false)
  def fromRowTensors(rows: Seq[DenseTensor1]): DenseDesignMatrix =
    fromRowArrays(rows.map(_.asArray))
  def concat(cs: BlasDenseTensor2*): DenseDesignMatrix =
    cs.map(DenseDesignMatrix(_)).reduceLeft(concat)
//  def concat(cs: DenseDesignMatrix*): DenseDesignMatrix =
//    cs.reduceLeft(concat)
  def concat(A: DenseDesignMatrix, B: DenseDesignMatrix): DenseDesignMatrix = {
    if (A.transposed || B.transposed) sys.error("unimpl")
    DenseDesignMatrix(BlasHelpers.blasDenseTensor2(BlasHelpers.concatColumns(A.mat.asArray, B.mat.asArray, A.mat.dim1, A.mat.dim2, B.mat.dim2), A.mat.dim1, A.mat.dim2 + B.mat.dim2))
  }
}

trait Labels[Label] {
  def labels: Seq[Label]
  def labeledInstances(designMatrix: DenseDesignMatrix): Seq[(DenseTensor1, Label)] =
    if (designMatrix.transposed) designMatrix.mat.cols().zip(labels.toSeq) else designMatrix.mat.rows().zip(labels.toSeq)
}

case class DoubleLabels(y: DenseTensor1) extends Labels[Double] {
  override def labels: Seq[Double] = y.asArray
}
case class MulticlassLabels(y: Array[Int]) extends Labels[Int] {
  def evaluateAccuracy(preds: BlasDenseTensor2): Double = {
    1.0 * preds.rows().zip(y).count({case (r, l) => r.maxIndex == l}) / y.size
  }
  override def labels: Seq[Int] = y
}

case class SparseDesignMatrix(X: Seq[SparseTensor with Tensor1], domainSize: Int, y: Option[DenseTensor1] = None, transposed: Boolean = false)

trait UnsupervisedFeatureMapTrainer[-A, +B] extends FeatureMapTrainer[A, B, Any] {
  outer =>
  def trainFeatureMap(X: A): FeatureMap[A, B]
  def trainFeatureMap(X: A, L: Any): FeatureMap[A, B] = trainFeatureMap(X)
  def +[C](trainer: UnsupervisedFeatureMapTrainer[B, C]): UnsupervisedFeatureMapTrainer[A, C] = UnsupervisedFeatureMapTrainer((X: A) => {
    val map1 = outer.trainFeatureMap(X)
    val X1 = map1.transform(X)
    val map2 = trainer.trainFeatureMap(X1)
    map1 + map2
  })
}
object UnsupervisedFeatureMapTrainer {
  def apply[A, B](fun: A => FeatureMap[A, B]): UnsupervisedFeatureMapTrainer[A, B] = new UnsupervisedFeatureMapTrainer[A, B] {
    override def trainFeatureMap(X: A): FeatureMap[A, B] = fun(X)
  }
}
object FeatureMapTrainer {
  def apply[A, B, Label](fun: (A, Label) => FeatureMap[A, B]): FeatureMapTrainer[A, B, Label] = new FeatureMapTrainer[A, B, Label] {
    override def trainFeatureMap(X: A, L: Label): FeatureMap[A, B] = fun(X, L)
  }
  def concat[A, B, Label](outer: FeatureMapTrainer[A, B, Label], trainer: FeatureMapTrainer[A, B, Label])
    (implicit ev: B =:= DenseDesignMatrix, ev2: DenseDesignMatrix =:= B): FeatureMapTrainer[A, B, Label] = FeatureMapTrainer((X: A, L: Label) => {
    val map1 = outer.trainFeatureMap(X, L)
    val map2 = trainer.trainFeatureMap(X, L)
    FeatureMap.concat(map1, map2)
  })
}

trait FeatureMapTrainer[-A, +B, Label] {
  outer =>
  def trainFeatureMap(X: A, L: Label): FeatureMap[A, B]
  def +[C](trainer: FeatureMapTrainer[B, C, Label]): FeatureMapTrainer[A, C, Label] = FeatureMapTrainer((X: A, L: Label) => {
    val map1 = outer.trainFeatureMap(X, L)
    val map2 = trainer.trainFeatureMap(map1.transform(X), L)
    map1 + map2
  })
}

object FeatureMap {
  def concat[A, B](outer: FeatureMap[A, B], map: FeatureMap[A, B])(implicit ev: B =:= DenseDesignMatrix, ev2: DenseDesignMatrix =:= B): FeatureMap[A, B] = new FeatureMap[A, B] {
    override def transform(X: A): B =
      DenseDesignMatrix.concat(outer.transform(X), map.transform(X))
  }
}

trait FeatureMap[-A, +B] {
  outer =>
  def transform(X: A): B
  // compose
  def +[C](map: FeatureMap[B, C]): FeatureMap[A, C] = new FeatureMap[A, C] {
    override def transform(X: A): C = map.transform(outer.transform(X))
  }
}

class DenseMatrixMultFeatureMap(val F: BlasDenseTensor2)
  extends DenseMatrixFeatureMap(designMatrix => {
    if (designMatrix.transposed) designMatrix.mat leftMultiply F else designMatrix.mat * F
  })

class DenseMatrixFeatureMap(val map: DenseDesignMatrix => BlasDenseTensor2) extends FeatureMap[DenseDesignMatrix, DenseDesignMatrix] {
  override def transform(designMatrix: DenseDesignMatrix): DenseDesignMatrix = {
    val mapped = map(designMatrix)
    designMatrix.copy(mat = mapped)
  }
}

object IdentityFeatureMap {
  def apply() = makeFeatureMapTrainer()
  def makeFeatureMapTrainer() = UnsupervisedFeatureMapTrainer((X: DenseDesignMatrix) => new DenseMatrixFeatureMap(_.mat))
}

object DensePCA {
  import BlasHelpers._
  // make a PCA feature map of rank k
  def apply(k: Int) = makeFeatureMapTrainer(k)
  def makeFeatureMapTrainer(k: Int) = UnsupervisedFeatureMapTrainer((X: DenseDesignMatrix) => makeFeatureMap(X, k))
  def makeFeatureMap(designMatrix: DenseDesignMatrix, k: Int): FeatureMap[DenseDesignMatrix, DenseDesignMatrix] = {
    val X = designMatrix.mat
    val V = eigenPSD(if (designMatrix.transposed) X multiplyTranspose X else X leftMultiply X, k)
    new DenseMatrixMultFeatureMap(V)
  }
  // get top k eigenvectors of dense sym def matrix
  // takes N x N matrix, returns N x k or x x N matrix of eigenvectors
  def eigenPSD(XTX: BlasDenseTensor2, k: Int, rowsAreEigenvectors: Boolean = false): BlasDenseTensor2 =
    eigenPSDValues(XTX, k, rowsAreEigenvectors)._1
  // returns (V,D) where V is matrix of right eigenvectors, D is diag matrix of eigenvalues
  def eigenPSDValues(XTX: BlasDenseTensor2, k: Int, rowsAreEigenvectors: Boolean = false): (BlasDenseTensor2, BlasDenseTensor2) = {
    val n = XTX.dim1

    val eigenvalues = doubleMatrix(n, 1)
    var v = doubleMatrix(n, n)
    val isuppz = new Array[Int](2 * n)

    // we are symmetric X^T X and we can use blas thing without worrying
    val copyX = BlasHelpers.transposedDoubleMatrix(XTX).dup()

    SimpleBlas.syevr('V', 'I', 'U', copyX, 0, 0, n - k + 1, n, 0, eigenvalues, v, isuppz)

    v = v.mmul(DoubleMatrix.diag(DoubleMatrix.ones(k), n, k))

    val V = if (rowsAreEigenvectors) transposedBlasTensor(v) else transposedBlasTensor(v).transpose()

    (V, BlasHelpers.diag(eigenvalues.data.take(k)))
  }

  def eigenPSD(X: BlasDenseTensor2): BlasDenseTensor2 =
    eigenPSD(X, X.dim1)
}

object RFF {
  def apply(d: Int, medianSampleSize: Int = 3000)(implicit random: Random) = makeFeatureMapTrainer(d, medianSampleSize)(random)
  def makeFeatureMapTrainer(d: Int, medianSampleSize: Int)(implicit random: Random) =
    UnsupervisedFeatureMapTrainer((designMatrix: DenseDesignMatrix) => makeFeatureMap(designMatrix.mat, d, medianSampleSize, designMatrix.transposed)(random))
  // for the version that doesnt cache the gaussian matrix, we need to like make a new RNG
  def makeFeatureMap(X: BlasDenseTensor2, d: Int, medianSampleSize: Int, transposedX: Boolean)(implicit random: Random): FeatureMap[DenseDesignMatrix, DenseDesignMatrix] = {
    val m = if (transposedX) X.dim2 else X.dim1
    val n = if (transposedX) X.dim1 else X.dim2
    // set bandwidth using median trick
    println("Estimating RBF kernel bandwidth...")
    val sampleSize = medianSampleSize
    val samples = {
      val featureVectors = if (transposedX) X.cols() else X.rows()
      (0 until m).sortBy(_ => random.nextDouble()).take(sampleSize).map(featureVectors(_))
    }
    val dists = new Array[Double](samples.size * samples.size)
    for (i <- 0 until samples.size; j <- 0 until samples.size; if i != j) {
      val s1 = samples(i)
      val s2 = samples(j)
      dists(i * samples.size + j) = (s1 - s2).twoNorm
    }
    java.util.Arrays.sort(dists)
    val median = dists(dists.size / 2)
    val scale = 1 / median

    println("Creating RFF feature map...")

    val b = BlasHelpers.rand(d)(random)
    b *= (2 * math.Pi)
    val R = BlasHelpers.randn(n, d)(random)
    R *= scale

    new DenseMatrixFeatureMap(designMatrix => {
      val X3 = if (designMatrix.transposed) designMatrix.mat leftMultiply R else designMatrix.mat * R
      if (designMatrix.transposed) X3 += (b outer BlasHelpers.ones(X3.dim2)) else X3 += (BlasHelpers.ones(X3.dim1) outer b)
      X3.applyFunction(math.cos(_))
      X3
    })
  }
}
