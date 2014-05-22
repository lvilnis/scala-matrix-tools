package md

import cc.factorie.la._
import scala.collection.mutable.ArrayBuffer
import java.io._
import java.util.zip.GZIPInputStream
import cc.factorie.app.classify.backend.LinearMulticlassClassifier
import cc.factorie.optimize._
import scala.util.Random
import cc.factorie.{maths, DiscreteDomain}
import org.jblas._
import scala.collection.JavaConverters._
import cc.factorie.model.{WeightsSet, WeightsMap}
import md.DoubleLabels
import md.MulticlassLabels

// todo:
// word convolution feature layer (and demo for conll 2003 chunking?)
// data loaders for sparse matrices
// Better design matrix variables with labels and domains and stuff? should use type members instead of parameters?
// sparse pca feature map
// Bayesian hyperparameter optimization w/ GPs

trait BatchMulticlassOptimizableObjective extends OptimizableObjective[BlasDenseTensor2, Array[Int]]

class LogBatchMulticlass extends BatchMulticlassOptimizableObjective {
  def valueAndGradient(prediction: BlasDenseTensor2, label: Array[Int]): (Double, BlasDenseTensor2) = {
    val normed = prediction
    BlasHelpers.expNormalizeRows(normed)

    val probs = BlasHelpers.indexRows(normed, label)
    // syntax for slicing out rows and applying functions to them?
    BlasHelpers.applyFunction(probs, math.log)
    val totalProb = BlasHelpers.sum(probs)
    normed *= -1.0
    BlasHelpers.addToRowIndices(normed, label, 1.0)
    (totalProb, normed)
  }
}

class SquaredBatchMulticlass extends BatchMulticlassOptimizableObjective {
  def valueAndGradient(prediction: BlasDenseTensor2, label: Array[Int]): (Double, BlasDenseTensor2) = {
    BlasHelpers.addToRowIndices(prediction, label, -1.0)
    prediction *= -2.0
    val sqNorms = BlasHelpers.twoNormSqRows(prediction)
    val totalValue = BlasHelpers.sum(sqNorms)
    (-totalValue / 4.0, prediction)

  }
}

object LoadMNIST {
  implicit val random = new Random(0)

  val mnistTrainPath = """../data/mnist/mnist"""
  val mnistTestPath = """../data/mnist/mnist.t"""

  val svmLightSeparatorRegex = """\s|:""".r
  val domainSize = 784
  val numClasses = 10

  def main(args: Array[String]): Unit = {

    val (mnistTrainMatrix, ytrain, mnistTestMatrix, ytest) = getTrainTestFeatures()

    val (mnistTrainSet, mnistTestSet) = (ytrain.labeledInstances(mnistTrainMatrix), ytest.labeledInstances(mnistTestMatrix))

    println("Start training...")

    //    val cls = new LinearMulticlassClassifier(numClasses, mnistTrainMatrix.domainSize) {
    //      override val weights = Weights(new BlasDenseTensor2(mnistTrainMatrix.domainSize, numClasses))
    //    }
    //

    val cls = new MLP.DenseBatchLinearLayer(mnistTrainMatrix.domainSize, numClasses)
    //    val cls = new MLP.DenseLinearLayer(mnistTrainMatrix.domainSize, numClasses)

    //    val cls = new MLP.DenseMultilayerPerceptron(mnistTrainMatrix.domainSize, 50, numClasses, 1, MLP.OptimizableActivations.tanh)
    //
    //    cls.initializeRandomly()


    // Wait, website says arrays are always copied on native calls?? wtf...

    // contract needs to be that predictions can be modified features place when they are passed features. when wouldn't this be the case?
    // this just returns sum of losses features the Double return value


    //    class HingeBatchMulticlass extends BatchMulticlassOptimizableObjective {
    //      def valueAndGradient(prediction: BlasDenseTensor2, label: Array[Int]): (Double, BlasDenseTensor2) = {
    //        val lossAugmented = prediction
    //        BlasHelpers.addToRowIndices(lossAugmented, label, -1.0)
    //
    //        val maxLabels = BlasHelpers.maxRowIndices(dt)
    //
    //        // ugh this isn't sparse - terrrible
    //        val values = BlasHelpers.indexRows(lossAugmented, label)
    //
    //
    //        BlasHelpers.expNormalizeRows(normed)
    //
    //        val probs = BlasHelpers.indexRows(normed, label)
    //        BlasHelpers.applyFunction(probs, math.log)
    //        val totalProb = BlasHelpers.sum(probs)
    //        normed *= -1.0
    //        BlasHelpers.addToRowIndices(normed, label, 1.0)
    //        (totalProb, normed)
    //      }
    //    }

    // TODO make sq loss batch thing... that prolly works really good


    val logBatchMultiClass = new LogBatchMulticlass
    val sqBatchMultiClass = new SquaredBatchMulticlass


    //    val examples = mnistTrainSet.map({case (f, l) => new MLP.MyPredictorExample(cls, f, l.toInt, OptimizableObjectives.logMulticlass)})

    // TODO make SGD thing that takes design matrix and row number

    val examples = Seq(new MLP.BatchLayerExample(cls, mnistTrainMatrix.mat, ytrain.labels.toArray, logBatchMultiClass, weight = 10.0))

    //    val examples = MiniBatchExample(50, mnistTrainSet.map({case (f, l) => new PredictorExample(cls, f, l.toInt, OptimizableObjectives.logMulticlass)}))

    class MyAdaGrad extends MyAdaptiveLearningRate
    class MyPerceptron extends MyConstantStepSize

    Trainer.batchTrain(cls.parameters, examples, useParallelTrainer = false, evaluate = () => {
      // make AdaGrad params variables so we can tweak it
      //    val startingRate = 0.0001
      //    val delta2 = 1
      //    val opt = new MyPerceptron { baseRate = startingRate / delta2 }   //new MyAdaGrad { rate = startingRate; delta = 1.0 }
      //    var i = 0
      //    Trainer.onlineTrain(cls.parameters, examples, maxIterations = 100, optimizer = opt, useParallelTrainer = false, evaluate = () => {
      //
      println("Train accuracy: ")
      //      val numTrainCorrect = mnistTrainSet.count({case (f, l) => l == cls.predict(f).maxIndex})
      //      println(numTrainCorrect * 1.0 / mnistTrainSet.size)
      println(ytrain.evaluateAccuracy(cls.predict(mnistTrainMatrix.mat)))

      println("Test accuracy: ")
      //      val numTestCorrect = mnistTestSet.count({case (f, l) => l == cls.predict(f).maxIndex})
      //      println(numTestCorrect * 1.0 / mnistTestSet.size)
      println(ytest.evaluateAccuracy(cls.predict(mnistTestMatrix.mat)))

      //      opt.reset()
      //      i += 1
      //      opt.baseRate = startingRate / (delta2 + i)
    })

    println("Train accuracy: ")
    //      val numTrainCorrect = mnistTrainSet.count({case (f, l) => l == cls.predict(f).maxIndex})
    //      println(numTrainCorrect * 1.0 / mnistTrainSet.size)
    println(ytrain.evaluateAccuracy(cls.predict(mnistTrainMatrix.mat)))

    println("Test accuracy: ")
    //      val numTestCorrect = mnistTestSet.count({case (f, l) => l == cls.predict(f).maxIndex})
    //      println(numTestCorrect * 1.0 / mnistTestSet.size)
    println(ytest.evaluateAccuracy(cls.predict(mnistTestMatrix.mat)))

  }

  def take(X: (DenseDesignMatrix, DoubleLabels), k: Int): (DenseDesignMatrix, DoubleLabels) = take(X._1, X._2, k)
  def shuffle(X: (DenseDesignMatrix, DoubleLabels))(implicit random: Random): (DenseDesignMatrix, DoubleLabels) = shuffle(X._1, X._2)(random)
  def take(X: DenseDesignMatrix, y: DoubleLabels, k: Int): (DenseDesignMatrix, DoubleLabels) = map(X, y, _.take(k))
  def shuffle(X: DenseDesignMatrix, y: DoubleLabels)(implicit random: Random): (DenseDesignMatrix, DoubleLabels) = map(X, y, _.sortBy(_ => random.nextDouble()))

  def map(X: DenseDesignMatrix, y: DoubleLabels, f: Seq[(DenseTensor1, Double)] => Seq[(DenseTensor1, Double)]): (DenseDesignMatrix, DoubleLabels) = {
    val instances = f(y.labeledInstances(X))
    val newX = if (X.transposed) BlasHelpers.columnArraysToBlasTensor(instances.map(_._1.asArray)) else BlasHelpers.rowArraysToBlasTensor(instances.map(_._1.asArray))
    (X.copy(mat = newX), DoubleLabels(new DenseTensor1(instances.map(_._2).toArray)))
  }

  def getTrainTestFeatures(): (DenseDesignMatrix, MulticlassLabels, DenseDesignMatrix, MulticlassLabels) = {
    println("Start reading data...")

    //    val (trainX, trainy) = take(shuffle(loadDenseLibSVM(mnistTrainPath, domainSize)), 10000)
    val (trainX, trainydoubles) = shuffle(loadDenseLibSVM(mnistTrainPath, domainSize))
    val (testX, testy) = loadDenseLibSVM(mnistTestPath, domainSize)

    val trainy = new MulticlassLabels(trainydoubles.y.asArray.map(_.toInt))

    trainX.mat *= 1.0 / 255
    testX.mat *= 1.0 / 255

    println("Start computing feature map...")

    //    val featureMapTrainer = IdentityFeatureMap() //+ DensePCA(k = 50) //+ RFF(d = 3000)
    //    val featureMapTrainer = DensePCA(k = 50).asInstanceOf[FeatureMapTrainer[DenseDesignMatrix, DenseDesignMatrix, MulticlassLabels]] + GEV(k = 10, n = 100) //+ RFF(d = 3000)
    val featureMapTrainer = GEV(k = 30, n = 100) //+ RFF(d = 3000)

    val finalMap = featureMapTrainer.trainFeatureMap(trainX, trainy)

    println("Applying feature map...")

    (finalMap.transform(trainX), trainy, finalMap.transform(testX), new MulticlassLabels(testy.y.asArray.map(_.toInt)))
  }

  def readFile(file: File, gzip: Boolean = false): BufferedReader = {
    val fileStream = new BufferedInputStream(new FileInputStream(file))
    new BufferedReader(new InputStreamReader(if (gzip) new GZIPInputStream(fileStream) else fileStream))
  }

  def loadDenseLibSVM(file: String, domainSize: Int): (DenseDesignMatrix, DoubleLabels) = {
    val instances = loadRowsLibSvm(file, domainSize)
    val X = BlasHelpers.rowArraysToBlasTensor(instances.map(_._1.asArray))
    val y = new DenseTensor1(instances.map(_._2).toArray)
    (DenseDesignMatrix(X, transposed = false), DoubleLabels(y))
    //    (DenseDesignMatrix(X.transpose(), transposed = true), DoubleLabels(y))
  }

  def loadRowsLibSvm(file: String, domainSize: Int): Seq[(DenseTensor1, Double)] = {
    val source = readFile(new java.io.File(file), gzip = false)
    val instances = ArrayBuffer[(DenseTensor1, Double)]()
    var lineNum = 0
    var line = null: String
    while ( {line = source.readLine(); line != null}) {
      lineNum += 1
      val fields = svmLightSeparatorRegex.split(line)
      val label = fields(0)
      val instance = new DenseTensor1(domainSize)
      var i = 1
      val len = fields.size
      while (i < len) {
        val idx = fields(i).toInt
        val value = fields(i + 1).toDouble
        instance +=(idx % domainSize, value)
        i += 2
      }
      instances += ((instance, label.toDouble))
    }
    instances
  }
}


// what is generalization of regression/classification? "complete interaction model"??

// TODO add assignment with masking with binary tensors
// pull out most array-specific stuff to generic helpers?
// can use update method update(tensor2, tensor2)
// look where 1st tensor2 is nonzero and copy second tensor2
// add slicing notation and features-place shuffle


class BlasDenseTensor2(values: Array[Double], dim1: Int, dim2: Int) extends DenseTensor2(dim1, dim2) {
  //  private var _transpose = false
  def this(dim1: Int, dim2: Int) = this(new Array[Double](dim1 * dim2), dim1, dim2)
  override def _initialArray: Array[Double] = values

  import BlasHelpers._

  override def copy: BlasDenseTensor2 = new BlasDenseTensor2(java.util.Arrays.copyOf(values, values.length), dim1, dim2)
  override def blankCopy: BlasDenseTensor2 = new BlasDenseTensor2(dim1, dim2)
  override def toTensor1: DenseTensor1 = new DenseTensor1(dim1 * dim2) {
    override def _initialArray: Array[Double] = values
  }

  def applyFunction(f: (Int, Double) => Double): Unit = {
    val len = values.size
    val arr = values
    var i = 0
    while (i < len) {
      val value = arr(i)
      arr(i) = f(i, value)
      i += 1
    }
  }

  def mapElements(f: Double => Double): BlasDenseTensor2 = {
    val res = copy
    res.applyFunction(f)
    res
  }

  def applyFunction(f: Double => Double): Unit =
    BlasHelpers.applyFunction(values, f)

  def rows(): Seq[DenseTensor1] =
    transposedDoubleMatrix(this).columnsAsList().iterator().asScala.toVector.map(c => new DenseTensor1(c.data))

  def cols(): Seq[DenseTensor1] =
    transposedDoubleMatrix(this).rowsAsList().iterator().asScala.toVector.map(c => new DenseTensor1(c.data))

  def transpose(): BlasDenseTensor2 = BlasHelpers.transpose(this)

  def \(t: BlasDenseTensor2): BlasDenseTensor2 =
    mldivide(this, t)

  def /(t: BlasDenseTensor2): BlasDenseTensor2 =
    mrdivide(this, t)

  def *(t: BlasDenseTensor2): BlasDenseTensor2 =
    mmult(this, t)

  // this doesn't transpose matrices like you would want it to
  // this should take row vectors, not col vectors
  // gives C = B^T A
  def leftMultiply(t: BlasDenseTensor2): BlasDenseTensor2 = mmult(t, this, transposedA = true)
  //  def leftMultiply(t: BlasDenseTensor2): BlasDenseTensor2 = mmult(t, this, transposedA = true)

  // gives C = B^T A^T
  def leftMultiplyTranspose(t: BlasDenseTensor2): BlasDenseTensor2 = mmult(t, this, transposedA = true, transposedB = true)

  // gives C = A B^T
  def multiplyTranspose(B: BlasDenseTensor2): BlasDenseTensor2 = mmult(this, B, transposedB = true)

  override def *(t: Tensor1): Tensor1 = t match {
    case t: DenseTensor1 =>

      val tArr = t.asArray
      val res = new DenseTensor1(dim1)
      val outArr = res.asArray
      val myDim1 = dim1
      val myDim2 = dim2
      val myArr = values
      var col = 0
      while (col < myDim2) {
        val v = tArr(col)
        if (v != 0.0) {
          var row = 0
          while (row < myDim1) {
            val offset = row * myDim2
            outArr(row) += (myArr(offset + col) * v)
            row += 1
          }
        }
        col += 1
      }
      res
    case _ => super.*(t)
  }
  override def leftMultiply(t: Tensor1): Tensor1 = t match {
    case t: DenseTensor1 =>
      // TODO copy this out to helper that works just with arrays and ints - since faster than BLAS to do m-v mult
      val tArr = t.asArray
      val res = new DenseTensor1(dim2)
      val outArr = res.asArray
      val myDim1 = dim1
      val myDim2 = dim2
      val myArr = values
      var i = 0
      while (i < myDim1) {
        val tv = tArr(i)
        if (tv != 0.0) {
          var j = 0
          while (j < myDim2) {
            outArr(j) += myArr(i * myDim2 + j) * tv
            j += 1
          }
        }
        i += 1
      }
      res
    case _ => super.leftMultiply(t)
  }
  // gives C = A B
  def *(B: Tensor2): Tensor2 = B match {
    case b: DenseTensor2 => mmult(this, b)
    case _ => sys.error("* unimplemented")
  }
  // gives C = A B^T
  def multiplyTranspose(B: Tensor2): Tensor2 = B match {
    case b: DenseTensor2 => mmult(this, b, transposedB = true)
    case _ => sys.error("multiply2Transpose unimplemented")
  }

  // gives C = B^T A^T
  def leftMultiplyTranspose(B: Tensor2): Tensor2 = B match {
    case b: DenseTensor2 => mmult(b, this, transposedB = true, transposedA = true)
    case _ => sys.error("multiply2Transpose unimplemented")
  }

  // gives C = B^T A
  def leftMultiply(B: Tensor2): Tensor2 = B match {
    case b: DenseTensor2 => mmult(b, this, transposedA = true)
    case _ => sys.error("leftMultiply unimplemented")
  }


  // should we have some sort of dense layered sparse binary or even singleton binary thing for labels?
  // that sounds like just another way to store a sparse binary tensor, but will require separate handling
  // we need to use it to dot rows for supervised stuff, or just mask things
  // but why do we need it? if we have the invariant that we need one thing features each row, then we know the i'th index is features the i'th row and just do i mod dim2
  // only problem is the invariant that it _must_ have one thing features each row makes it hard to modify - basically immutable.
  // OK so it will be immutable, that makes sense

  // try using jblas builtin functions

  //  def apply(i: SparseBinaryTensor): SparseIndexedTensor = {
  //    val ret = new SparseIndexedTensor2(dim1, dim2)
  //    i.foreachActiveElement((idx, _) => ret += (idx, values(idx)))
  //    ret
  //  }

  // a bunch of methods for updating only certain indices, adding to them, etc
  def update(i: SparseBinaryTensor, v: Double): Unit = {
    i._makeReadable()
    val myArr = values
    val indices = i._indices
    val activeDomainSize = i.activeDomainSize
    var j = 0
    while (j < activeDomainSize) {
      myArr(indices(j)) = v
      j += 1
    }
  }
  // also, I should have an overload of "update" that takes a arbitrary function to replace "applyFunction"?

  // should have something to check for outers and uniforms and stuff? nah that'll get handled pretty well actually
  def update(i: SparseBinaryTensor, v: Tensor): Unit = {
    i._makeReadable()
    val myArr = values
    val indices = i._indices
    val activeDomainSize = i.activeDomainSize
    var j = 0
    while (j < activeDomainSize) {
      val idx = indices(j)
      myArr(idx) = v(idx)
      j += 1
    }
  }

  // add method that takes a predicate and returns 1's where the predicate is true
  // but why not also have an "update" method that takes a predicate directly?

  def update(i: Double => Boolean, v: Double): Unit = {
    val arr = values
    val len = arr.length
    var j = 0
    while (j < len) {
      if (i(arr(j))) arr(j) = v
      j += 1
    }
  }


  // should we allow this for not sparse binary tensors? what use would that serve features practice vs just casting...
  // what about if we do like "eq 0" to get a mask thats mostly 1's?
  //  def update(i: Tensor, v: Double): Unit = values(i) = v


}


// This tensor is immutable
//class DenseLayeredSingletonBinaryTensor2(val _hotIndicesUnsafe: Array[Int], override val dim1: Int, override val dim2: Int) extends Tensor2 with SparseBinaryTensor {
//  override def blankCopy: DenseLayeredSingletonBinaryTensor2 = new DenseLayeredSingletonBinaryTensor2(dim1, dim2)
//  override def copy: DenseLayeredSingletonBinaryTensor2 = new DenseLayeredSingletonBinaryTensor2(dim1, dim2)
//
//}


object BlasHelpers {

  // make one method parametrized by UpperTriangular, LowerTriangular, Symmetric, Standard, and a transpose option and stuff?
  def trmldivide(A: BlasDenseTensor2, B: BlasDenseTensor2, upperTriangular: Boolean = true): BlasDenseTensor2 =
    if (upperTriangular) uppertrmlDivideBackend(A, B) else lowertrmlDivideBackend(A, B)

  // trsm?
  // Solve system of linear equations AX = B for X
  // assuming A is upper triangular
//  private def trmlDivideBackend(A: BlasDenseTensor2, B: BlasDenseTensor2): BlasDenseTensor2 = {
//    //
//    assert(A.dim1 == A.dim2)
//    val d = A.dim1
//    val X = new BlasDenseTensor2(A.dim1, B.dim2)
//    for (h <- 0 until B.dim2; k <- d - 1 to 0 by -1) {
//      var sum = 0.0
//      var i = 0
//      val len = d - 1 - k
//      while (i < len) {
//        sum += A(k, i) * X(i)
//        i += 1
//      }
//      val thingie = (sum + B(k, h)) / A(k, k)
//      println(thingie)
//      X(k, h) = thingie
//    }
//    X
//  }
  def uppertrmlDivideBackend(A: BlasDenseTensor2, B: BlasDenseTensor2): BlasDenseTensor2 = {
    assert(A.dim1 == A.dim2)
    val d = A.dim1
    val X = new BlasDenseTensor2(A.dim1, B.dim2)
    for (h <- 0 until B.dim2; k <- d - 1 to 0 by -1) {
      var sum = 0.0
      var i = 0
      val len = d - 1 - k
      while (i < len) {
        sum += A(k, i) * X(i, h)
        i += 1
      }
      val thingie = (sum + B(k, h)) / A(k, k)
//      println(thingie)
      X(k, h) = thingie
    }
    X
  }
  def lowertrmlDivideBackend(A: BlasDenseTensor2, B: BlasDenseTensor2): BlasDenseTensor2 = {
    //
    assert(A.dim1 == A.dim2)
    val d = A.dim1
    val X = new BlasDenseTensor2(A.dim1, B.dim2)
    for (h <- 0 until B.dim2; k <- 0 until d) {
      var sum = 0.0
      var i = 0
      val len = k
      while (i < len) {
        sum += A(k, i) * X(i, h)
        i += 1
      }
      val thingie = (sum + B(k, h)) / A(k, k)
//      println(thingie)
      X(k, h) = thingie
    }
    X
  }
  def trmrdivide(A: BlasDenseTensor2, B: BlasDenseTensor2, transposeA: Boolean = false): BlasDenseTensor2 =
    trmldivide(B.transpose(), A.transpose(), transposeA).transpose()

  // Solve system of linear equations Ax = B for x
  // equiv to matlab A\B
  // TODO need to make this more efficient
  def mldivide(A: BlasDenseTensor2, B: BlasDenseTensor2): BlasDenseTensor2 =
    transposedBlasTensor(Solve.solveSymmetric(transposedDoubleMatrix(A).transpose(), transposedDoubleMatrix(B).transpose()).transpose())

  // Solve system of linear equations xA = B for x
  // same as A^T x^T = B^T for x^T
  def mrdivide(A: BlasDenseTensor2, B: BlasDenseTensor2): BlasDenseTensor2 =
    transposedBlasTensor(Solve.solveSymmetric(transposedDoubleMatrix(B), transposedDoubleMatrix(A)))

  def cholesky(dt: BlasDenseTensor2): BlasDenseTensor2 =
    transposedBlasTensor(Decompose.cholesky(transposedDoubleMatrix(dt)))

  def sumLogProbs(vals: Array[Double], offset: Int, length: Int, stride: Int = 1): Double = {
    val LOGTOLERANCE = 30.0

    var max = vals(offset)
    var maxIdx = offset
    var i = 1
    while (i < length) {
      val idx = offset + stride * i
      val v = vals(idx)
      if (v > max) {
        max = v
        maxIdx = idx
      }
      i += 1
    }
    var anyAdded = false
    var intermediate = 0.0
    val cutoff = max - LOGTOLERANCE
    i = 0
    while (i < length) {
      val idx = offset + stride * i
      if (vals(idx) >= cutoff && idx != maxIdx) {
        anyAdded = true
        intermediate += math.exp(vals(idx) - max)
      }
      i += 1
    }
    if (anyAdded)
      max + math.log1p(intermediate)
    else
      max
  }

  def expNormalize(vals: Array[Double], offset: Int, length: Int, stride: Int = 1): Double = {
    val sum = sumLogProbs(vals, offset, length, stride)
    var i = 0
    while (i < length) {
      val idx = offset + stride * i
      vals(idx) = math.exp(vals(idx) - sum)
      i += 1
    }
    sum
  }

  // assumes these are stored in row-major order
  def concatColumns(A: Array[Double], B: Array[Double], dim1: Int, dim2a: Int, dim2b: Int): Array[Double] = {
    val dim2 = dim2a + dim2b
    val ret = new Array[Double](dim1 * dim2)
    var i = 0
    while (i < dim1) {
      var j = 0
      while (j < dim2a) {
        ret(dim2 * i + j) = A(dim2a * i + j)
        j += 1
      }
      j = 0
      while (j < dim2b) {
        ret(dim2 * i + dim2a + j) = B(dim2b * i + j)
        j += 1
      }
      i += 1
    }
    ret
  }

  def maxIndex(vals: Array[Double], offset: Int, length: Int, stride: Int = 1): Int = {
    var max = vals(offset)
    var maxIdx = 0
    var i = 1
    while (i < length) {
      val idx = offset + stride * i
      val v = vals(idx)
      if (v > max) {
        max = v
        maxIdx = i
      }
      i += 1
    }
    maxIdx
  }

  def maxRowIndices(dt: BlasDenseTensor2): Array[Int] = {
    val output = new Array[Int](dt.dim1)
    val dtArr = dt.asArray
    val dim1 = dt.dim1
    val dim2 = dt.dim2
    var i = 0
    while (i < dim1) {
      output(i) = maxIndex(dtArr, i * dim2, dim2)
      i += 1
    }
    output
  }

  def twoNormSq(vals: Array[Double], offset: Int, length: Int, stride: Int = 1): Double = {
    var sum = 0.0
    var i = 0
    while (i < length) {
      val idx = offset + stride * i
      sum += vals(idx) * vals(idx)
      i += 1
    }
    sum
  }

  def twoNormSqRows(dt: BlasDenseTensor2): Array[Double] = {
    val output = new Array[Double](dt.dim1)
    val dtArr = dt.asArray
    val dim1 = dt.dim1
    val dim2 = dt.dim2
    var i = 0
    while (i < dim1) {
      output(i) = twoNormSq(dtArr, i * dim2, dim2)
      i += 1
    }
    output
  }


  def expNormalizeRows(dt: BlasDenseTensor2): Array[Double] = {
    val output = new Array[Double](dt.dim1)
    val dtArr = dt.asArray
    val dim1 = dt.dim1
    val dim2 = dt.dim2
    var i = 0
    while (i < dim1) {
      output(i) = expNormalize(dtArr, i * dim2, dim2)
      i += 1
    }
    output
  }

  def sum(arr: Array[Double]): Double = {
    val len = arr.size
    var sum = 0.0
    var i = 0
    while (i < len) {
      sum += arr(i)
      i += 1
    }
    sum
  }

  def applyFunction(arr: Array[Double], f: Double => Double): Unit = {
    val len = arr.size
    var i = 0
    while (i < len) {
      arr(i) = f(arr(i))
      i += 1
    }
  }

  def addToRowIndices(dt: BlasDenseTensor2, indices: Array[Int], toAdd: Double): Unit = {
    val dtArr = dt.asArray
    val dim1 = dt.dim1
    val dim2 = dt.dim2
    var i = 0
    while (i < dim1) {
      val idx = i * dim2 + indices(i)
      dtArr(idx) += toAdd
      i += 1
    }
  }

  // need to get addition of arrays and stuff... all tensors should just call on this
  // or should we just do it with jblas for some things - vector stuff seems borderline

  def filterRows(dt: BlasDenseTensor2, indices: Array[Int]): BlasDenseTensor2 = {
    val output = new BlasDenseTensor2(indices.size, dt.dim2)
    val dtArr = dt.asArray
    val outDim1 = output.dim1
    val dim2 = dt.dim2
    var i = 0
    while (i < outDim1) {
      var j = 0
      while (j < dim2) {
        output(i * dim2 + j) = dtArr(indices(i) * dim2 + j)
        j += 1
      }
      i += 1
    }
    output
  }

  // could "indices" just be a SparseBinaryTensor2 ??
  // sure.. or a seq of SparseBinaryTensor1 or SingletonBinaryTensor1
  // we really need something where each layer is a singleton, not just one singleton layer - those are just outer products
  // we'll use Array[Int] for now
  def indexRows(dt: BlasDenseTensor2, indices: Array[Int]): Array[Double] = {
    val output = new Array[Double](dt.dim1)
    val dtArr = dt.asArray
    val dim1 = dt.dim1
    val dim2 = dt.dim2
    var i = 0
    while (i < dim1) {
      output(i) = dtArr(i * dim2 + indices(i))
      i += 1
    }
    output
  }

  //  def dotRows(dt1: BlasDenseTensor2, dt2: BlasDenseTensor2): Array[Double] = {
  //
  //  }
  //
  def blasColumnDenseTensor2(dt: DenseTensor1): BlasDenseTensor2 = new BlasDenseTensor2(dt.asArray, dt.dim1, 1)
  def blasRowDenseTensor2(dt: DenseTensor1): BlasDenseTensor2 = new BlasDenseTensor2(dt.asArray, 1, dt.dim1)
  def transpose(bdt: DenseTensor2): BlasDenseTensor2 = {
    val dim1 = bdt.dim1
    val dim2 = bdt.dim2
    if (dim1 == 1 || dim2 == 1) new BlasDenseTensor2(bdt.asArray, dim2, dim1)
    else transposedBlasTensor(transposedDoubleMatrix(bdt).transpose())
  }

  def blasDenseTensor2(dt: DenseTensor2): BlasDenseTensor2 = new BlasDenseTensor2(dt.asArray, dt.dim1, dt.dim2)
  def blasDenseTensor2(values: Array[Double], dim1: Int, dim2: Int): BlasDenseTensor2 = new BlasDenseTensor2(values, dim1, dim2)
  def blasDenseTensor2(dim1: Int, dim2: Int): BlasDenseTensor2 = new BlasDenseTensor2(dim1, dim2)
  def shuffleRows(t: BlasDenseTensor2)(implicit random: Random): BlasDenseTensor2 =
    rowArraysToBlasTensor(t.rows().sortBy(_ => random.nextDouble()).map(_.asArray))
  def takeRows(t: BlasDenseTensor2, k: Int): BlasDenseTensor2 = diag(ones(k), k, t.dim1) * t


  def rowArraysToBlasTensor(rows: Seq[Array[Double]]): BlasDenseTensor2 = {
    if (rows.isEmpty) sys.error("empty!")
    val numRows = rows.size
    val numCols = rows(0).size
    val arr = new Array[Double](numRows * numCols)
    var i = 0
    while (i < numRows) {
      System.arraycopy(rows(i), 0, arr, i * numCols, numCols)
      i += 1
    }
    blasDenseTensor2(arr, numRows, numCols)
  }
  def columnArraysToBlasTensor(columns: Seq[Array[Double]]): BlasDenseTensor2 = {
    if (columns.isEmpty) sys.error("empty!")
    val numRows = columns(0).size
    val numCols = columns.size
    val arr = new Array[Double](numRows * numCols)
    var j = 0
    while (j < numCols) {
      val curCol = columns(j)
      var i = 0
      while (i < numRows) {
        arr(i * numCols + j) = curCol(i)
        i += 1
      }
      j += 1
    }
    blasDenseTensor2(arr, numRows, numCols)
  }

  def columnTensorsToBlasTensor(columns: Seq[DenseTensor1]): BlasDenseTensor2 = columnArraysToBlasTensor(columns.map(_.asArray))
  def rowTensorsToBlasDenseTensor2(rows: Seq[DenseTensor1]): BlasDenseTensor2 = rowArraysToBlasTensor(rows.map(_.asArray))

  def mmult(A: DenseTensor2, B: DenseTensor2, transposedA: Boolean = false, transposedB: Boolean = false): BlasDenseTensor2 = {
    val res1 = mmultBackend(A, B, transposedA, transposedB)
    //    val res2 = mmultBackend(A, blasDenseTensor2(B).transpose(), transposedA, !transposedB)
    //    val res3 = mmultBackend(blasDenseTensor2(A).transpose(), B, !transposedA, transposedB)
    //    val res4 = mmultBackend(blasDenseTensor2(A).transpose(), blasDenseTensor2(B).transpose(), !transposedA, !transposedB)
    //    val res5 = transposedBlasTensor( (if (transposedA) transposedDoubleMatrix(A) else transposedDoubleMatrix(A).transpose() )
    //      .mmul(if (transposedB) transposedDoubleMatrix(B) else transposedDoubleMatrix(B).transpose()).transpose())
    //
    //
    //    val n1 = res1.twoNorm
    //    val n2 = res2.twoNorm
    //    val n3 = res3.twoNorm
    //    val n4 = res4.twoNorm
    //    val n5 = res5.twoNorm
    //
    //    if (Seq(n1, n2, n3, n4, n5).distinct.length > 1)
    //      println("ahh!")
    res1
  }

  private def mmultBackend(A: DenseTensor2, B: DenseTensor2, transposedA: Boolean, transposedB: Boolean): BlasDenseTensor2 = {
    val alpha = 1.0
    val beta = 0.0
    val matAT = transposedDoubleMatrix(A)
    val matBT = transposedDoubleMatrix(B)
    // C = A B
    // C^T = B^T A^T
    if (!transposedA && !transposedB) {
      //      println("1")
      val m = B.dim2
      val n = A.dim1
      val k = A.dim2
      val matCT = doubleMatrix(m, n)
      if (true) {
        matBT.mmuli(matAT, matCT)
      } else {
        NativeBlas.dgemm('N', 'N', m, n, k, alpha, matBT.data, 0, m, matAT.data, 0, k, beta, matCT.data, 0, m)
      }
      transposedBlasTensor(matCT)
    } else if (transposedA && transposedB) {
      // C = A^T B^T
      // C^T = B A
      //      println("2")
      val m = B.dim1
      val n = A.dim2
      val k = A.dim1
      val matCT = doubleMatrix(m, n)
      NativeBlas.dgemm('T', 'T', m, n, k, alpha, matBT.data, 0, k, matAT.data, 0, n, beta, matCT.data, 0, m)
      transposedBlasTensor(matCT)
    } else if (transposedA) {
      // C = A^T B
      // C^T = B^T A
      //      println("3")
      val m = B.dim2
      val n = A.dim2
      val k = A.dim1
      val matCT = doubleMatrix(m, n)
      NativeBlas.dgemm('N', 'T', m, n, k, alpha, matBT.data, 0, m, matAT.data, 0, n, beta, matCT.data, 0, m)
      transposedBlasTensor(matCT)
    } else /*if (transposedB)*/ {
      // C = A B^T
      // C^T = B A^T
      //      println("4")
      val m = B.dim1
      val n = A.dim1
      val k = A.dim2
      val matCT = doubleMatrix(m, n)
      NativeBlas.dgemm('T', 'N', m, n, k, alpha, matBT.data, 0, k, matAT.data, 0, k, beta, matCT.data, 0, m)
      transposedBlasTensor(matCT)
    }
  }

  def randfill(arr: Array[Double])(implicit random: Random): Unit = {
    val len = arr.size
    var i = 0
    while (i < len) {
      arr(i) = random.nextDouble()
      i += 1
    }
  }
  def randnfill(arr: Array[Double])(implicit random: Random): Unit = {
    val len = arr.size
    var i = 0
    while (i < len) {
      arr(i) = random.nextGaussian()
      i += 1
    }
  }

  def rand(n: Int)(implicit random: Random): DenseTensor1 = processArray(new DenseTensor1(n), randfill(_)(random))
  def rand(rows: Int, cols: Int)(implicit random: Random): BlasDenseTensor2 = processArray(new BlasDenseTensor2(rows, cols), randfill(_)(random))

  def randn(n: Int)(implicit random: Random): DenseTensor1 = processArray(new DenseTensor1(n), randnfill(_)(random))
  def randn(rows: Int, cols: Int)(implicit random: Random): BlasDenseTensor2 = processArray(new BlasDenseTensor2(rows, cols), randnfill(_)(random))

  def processArray[U <: DenseTensor](dt: U, action: Array[Double] => Unit): U = {
    action(dt.asArray)
    dt
  }

  def eye(n: Int): BlasDenseTensor2 = diag(ones(n))
  def ones(n: Int): DenseTensor1 = new DenseTensor1(n, 1.0)
  def ones(rows: Int, cols: Int): BlasDenseTensor2 = {
    val res = new BlasDenseTensor2(rows, cols)
    val resArr = res.asArray
    java.util.Arrays.fill(resArr, 1.0)
    res
  }

  def diag(t: DenseTensor1, rows: Int, cols: Int): BlasDenseTensor2 = diag(t.asArray, rows, cols)
  def diag(t: DenseTensor1): BlasDenseTensor2 = diag(t.asArray)
  def diag(arr: Array[Double]): BlasDenseTensor2 = diag(arr, arr.size, arr.size)
  def diag(arr: Array[Double], rows: Int, cols: Int): BlasDenseTensor2 = {
    val len = arr.size
    val res = blasDenseTensor2(rows, cols)
    val resArr = res.asArray
    var i = 0
    while (i < len) {
      resArr(cols * i + i) = arr(i)
      i += 1
    }
    res
  }

  def transposedBlasTensor(dm: DoubleMatrix): BlasDenseTensor2 = blasDenseTensor2(dm.data, dm.columns, dm.rows)
  def transposedDoubleMatrix(dt: DenseTensor2): DoubleMatrix = doubleMatrix(dt.asArray, dt.dim2, dt.dim1)
  def doubleMatrix(values: Array[Double], rows: Int, cols: Int): DoubleMatrix = {
    val AT = new DoubleMatrix()
    AT.data = values
    AT.length = rows * cols
    AT.rows = rows
    AT.columns = cols
    AT
  }
  def doubleMatrix(rows: Int, cols: Int): DoubleMatrix =
    new DoubleMatrix(rows, cols)
}

object TensorHelpers {
  def applyFunction(t: Tensor, f: Double => Double): Unit = t match {
    case t: DenseTensor =>
      val tArr = t.asArray
      val len = t.size
      var i = 0
      while (i < len) {
        tArr(i) = f(tArr(i))
        i += 1
      }
  }
}

trait MyConstantStepSize extends GradientStep {
  var baseRate = 1.0
  override def lRate(weights: WeightsSet, gradient: WeightsMap, value: Double): Double = baseRate
}


trait MyAdaptiveLearningRate extends GradientStep {
  /**
   * The base learning rate
   */
  var rate: Double = 1.0
  /**
   * The learning rate decay factor.
   */
  var delta: Double = 0.1
  private var HSq: WeightsMap = null
  var printed = false
  override def initializeWeights(weights: WeightsSet) {
    super.initializeWeights(weights)
    if (HSq == null) HSq = weights.blankDenseMap
  }
  override def reset(): Unit = {
    super.reset()
    HSq = null
  }
  override def processGradient(weights: WeightsSet, gradient: WeightsMap): Unit = {
    val eta = rate
    //    val l2 = 0.1
    //    gradient += (weightsSet, -l2)
    if (HSq == null) HSq = weights.blankDenseMap
    for (template <- gradient.keys) {
      gradient(template) match {
        case t: Outer1Tensor2 if t.tensor1.isDense && t.tensor2.isDense =>
          gradient(template) = new DenseTensor2(t.dim1, t.dim2)
          gradient(template) += t
        case t: Outer1Tensor2 =>
          gradient(template) = new SparseIndexedTensor2(t.dim1, t.dim2)
          gradient(template) += t
        case t: SparseBinaryTensor1 =>
          gradient(template) = new SparseIndexedTensor1(t.dim1)
          gradient(template) += t
        case t: SparseBinaryTensor2 =>
          gradient(template) = new SparseIndexedTensor2(t.dim1, t.dim2)
          gradient(template) += t
        case t: SparseBinaryTensor3 =>
          gradient(template) = new SparseIndexedTensor3(t.dim1, t.dim2, t.dim3)
          gradient(template) += t
        case t: SparseBinaryTensor4 =>
          gradient(template) = new SparseIndexedTensor4(t.dim1, t.dim2, t.dim3, t.dim4)
          gradient(template) += t
        case _ =>
      }
    }
    for (template <- gradient.keys)
      (gradient(template), HSq(template)) match {
        case (g: DenseTensor, hSq: DenseTensor) =>
          //          println(hSq)
          val gArr = g.asArray
          val hArr = hSq.asArray
          var i = 0
          val len = gArr.length
          while (i < len) {
            if (gArr(i) != 0) {
              hArr(i) += gArr(i) * gArr(i)
              val h = math.sqrt(hArr(i)) + delta
              val t1 = eta / h
              gArr(i) *= t1
              //              assert(!gArr(i).isNaN)
            }
            i += 1
          }
        case (g: SparseIndexedTensor, hSq: DenseTensor) =>
          val hArr = hSq.asArray
          var i = 0
          val len = g.activeDomainSize
          val indices = g._indices
          val values = g._values
          while (i < len) {
            val g = values(i)
            if (g != 0) {
              val idx = indices(i)
              hArr(idx) += g * g
              val h = math.sqrt(hArr(idx)) + delta
              val t1 = eta / h
              values(i) *= t1
              //              assert(!values(i).isNaN)
            }
            i += 1
          }
        case (g: SparseIndexedTensor, hSq: Tensor) =>
          if (!printed) {
            printed = true
            println("No implementations for: " + weights(template).getClass.getName + " " +
                    gradient(template).getClass.getName + " " + HSq(template).getClass.getName)
          }
          var i = 0
          val len = g.activeDomainSize
          val indices = g._indices
          val values = g._values
          while (i < len) {
            val g = values(i)
            if (g != 0) {
              val idx = indices(i)
              hSq(idx) += g * g
              val h = math.sqrt(hSq(idx)) + delta
              val t1 = eta / h
              values(i) *= t1
              //              assert(!values(i).isNaN)
            }
            i += 1
          }
      }
  }
}