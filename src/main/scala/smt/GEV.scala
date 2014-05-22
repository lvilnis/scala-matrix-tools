package md

import org.jblas.{SimpleBlas, DoubleMatrix}
import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashMap
import scala.collection.mutable

object GEV {
  def apply(k: Int, n: Int = 50, threshold: Double = 2.5, gamma: Double = 0.5, disallowedPairs: Set[(Int, Int)] = Set())(implicit random: scala.util.Random) =
    makeFeatureMapTrainer(k, n, threshold, gamma, disallowedPairs)
  def makeFeatureMapTrainer(k: Int, n: Int, threshold: Double, gamma: Double, disallowedPairs: Set[(Int, Int)] = Set())(implicit random: scala.util.Random) =
    FeatureMapTrainer((X: DenseDesignMatrix, Y: MulticlassLabels) => makeFeatureMap(X, Y, k, n, threshold, gamma, disallowedPairs))
  def makeFeatureMap(designMatrix: DenseDesignMatrix, labels: MulticlassLabels, k: Int, n: Int, threshold: Double, gamma: Double, disallowedPairs: Set[(Int, Int)])(implicit random: scala.util.Random): FeatureMap[DenseDesignMatrix, DenseDesignMatrix] = {
    val X = designMatrix.mat
    val numClasses = labels.y.distinct.size

    val classesAndRows = labels.y.zipWithIndex.groupBy(_._1).mapValues(_.map(_._2)).view.force

    val classesAndCovariances = classesAndRows
      .map({case (c, rows) => (c, BlasHelpers.filterRows(X, rows))})
      .map({case (c, x) => (c, x leftMultiply x)})
      .toMap

    val chols = new HashMap[Int, BlasDenseTensor2]
    val cholts = new HashMap[Int, BlasDenseTensor2]
    val vectors = new ArrayBuffer[Array[Double]]
    val pairs = generateClassPairingsSkiplist(numClasses, n, disallowedPairs)
    for ((c1, c2) <- pairs; cov1 <- classesAndCovariances.get(c1); cov2 <- classesAndCovariances.get(c2)) {
      val GEVResults(vecs, vals) = findGEVs(cov1, cov2, c1, c2, k, chols, cholts, gamma)
      println(s"c1: $c1 c2: $c2")
//      val GEVResults(vecs, vals) = findGEVsBlas(cov1, cov2, k, gamma)
      val toAdd = vecs.zip(vals).filter({
        case (vec, value) =>
          println(s"gev: $value")
//          true
          value >= threshold
      }).map(_._1)
      val batch = toAdd.size
      vectors ++= toAdd
    }

    val mat = BlasHelpers.columnArraysToBlasTensor(vectors)

    new DenseMatrixFeatureMap(designMatrix => {
      val X3 = if (designMatrix.transposed) designMatrix.mat leftMultiply mat else designMatrix.mat * mat
//      val pos = X3
//      val neg = X3.copy
//      pos.applyFunction(x => math.sqrt(1 + math.max(0, x)) - 1)
//      neg.applyFunction(x => math.sqrt(1 + math.max(0, -x)) - 1)
//      /*DenseDesignMatrix.concat(designMatrix, */DenseDesignMatrix.concat(DenseDesignMatrix(pos), DenseDesignMatrix(neg) ).mat
//      DenseDesignMatrix.concat(DenseDesignMatrix(pos), DenseDesignMatrix(neg)).mat


      // trsv triangular solve

      DenseDesignMatrix.concat(
        designMatrix.mat,
        X3.mapElements(x => math.sqrt(1 + math.max(0, x)) - 1),
        X3.mapElements(x => math.sqrt(1 + math.max(0, -x)) - 1)
//        X3.mapElements(x => math.max(0, x)),
//        X3.mapElements(x => math.max(0, -x))
//        X3.mapElements(x => math.pow(1 + math.max(0, x) - 1, 1.5)),
//        X3.mapElements(x => math.pow(1 + math.max(0, -x) - 1, 1.5))
      ).mat
    })
//
//    for (i <- 0 until numClasses; j <- 0 until i) {
//      val cov1 = classesAndCovariances(i)
//      val cov2 = classesAndCovariances(j)
//      findGEVs(cov1, cov2, k)
//    }
//
//    // now we need to put all the gev's in a big matrix so we can compute the feature map by matrix multiplication
//    // then apply a nonlinearity
////    for ((c1, cov1) <- classesAndCovariances; (c2, cov2) <- classesAndCovariances)
//
//    new DenseMatrixFeatureMap(designMatrix => {
//      val X3 = if (designMatrix.transposed) designMatrix.mat leftMultiply R else designMatrix.mat * R
//      if (designMatrix.transposed) X3 += (b outer BlasHelpers.ones(X3.dim2)) else X3 += (BlasHelpers.ones(X3.dim1) outer b)
//      X3.applyFunction(math.cos(_))
//      X3
//    })

  }

  // make a hypercube
  // A_n     =     A_n           I_{2^{n-1}}
  //               I_{2^{n-1}}   A_n
  // write some cool sparse tile functions to do this in linear algebra (or even kronecker product)
  // OK so how do we ensure everything is connected? BFS or DFS of graph and then take the first "numClasses" things
  object Hypercube {
    class Node(val id: Int) {
      val neighbors = ArrayBuffer[Int]()
    }
    def makeHypercube(d: Int): Seq[Hypercube.Node] = {
      var id = 0
      def nextId(): Int = {id += 1; id}
      def newNode(): Node = new Node(nextId())
      def copyGraph(graph: Seq[Node]): Seq[Node] = {
        val newNodes = Seq.fill(graph.size)(newNode())
        val idMap = graph.map(_.id).zip(newNodes.map(_.id)).toMap
        for ((node, newNode) <- graph.zip(newNodes))
          newNode.neighbors ++= node.neighbors.map(idMap)
        newNodes
      }
      val nodes = ArrayBuffer[Node](newNode())
      for (_ <- 0 until d) {
        val graphCopy = copyGraph(nodes)
        for ((n1, n2) <- nodes.zip(graphCopy)) {
          n1.neighbors += n2.id
          n2.neighbors += n1.id
        }
        nodes ++= graphCopy
      }
      nodes
    }
  }

  // sparsetensor2?
  def generateClassPairings(numClasses: Int, n: Int)(implicit random: scala.util.Random): Seq[(Int, Int)] = {
    (0 until numClasses).flatMap(i => (0 until numClasses).map(j => (i, j))).filter({case (i, j) => i != j}).sortBy(_ => random.nextDouble()).take(n)
  }

  // takes nlogn things then randomly samples
  def generateClassPairingsSkiplist(numClasses: Int, n: Int, disallowedPairs: Set[(Int, Int)])(implicit random: scala.util.Random): Seq[(Int, Int)] = {

    val classes = (0 until numClasses).toSeq
    val layers = new ArrayBuffer[Seq[Int]] += classes
    val logn = math.ceil(math.log(numClasses)).toInt

    val p = 0.5

    for (_ <- 0 until logn) {
      val curLayer = new ArrayBuffer[Int]
      for (prevLayerElement <- layers.last) {
        if (random.nextDouble() < p)
          curLayer += prevLayerElement
      }
      layers += curLayer
    }

    val pairs = layers.filter(_.size > 1).flatMap(layer => layer.sliding(2).toSeq.map(s => (s(0), s(1))))
    val symmetricPairs = pairs ++ pairs.map(p => (p._2, p._1))

    symmetricPairs.sortBy(_ => random.nextDouble()).filterNot(disallowedPairs).take(n)
  }

  // what order does this return things in?
  // solve Bx = lambda Ax
  case class GEVResults(vectors: List[Array[Double]], values: List[Double])
  def findGEVs(A: BlasDenseTensor2, Bin: BlasDenseTensor2, ai: Int, bi: Int, k: Int, chols: HashMap[Int, BlasDenseTensor2], cholts: HashMap[Int, BlasDenseTensor2], gamma: Double): GEVResults = {
    // add a ridge
    val avgEig = Bin.trace / Bin.dim1
    val B = Bin.copy
    B += (BlasHelpers.eye(Bin.dim1), avgEig * gamma)

    val lt = cholts.getOrElseUpdate(bi, BlasHelpers.cholesky(B).transpose())
    val l = chols.getOrElseUpdate(bi, cholts(bi).transpose())


//        val C = BlasHelpers.trmldivide(lt.transpose(), BlasHelpers.trmldivide(l, A, upperTriangular = false).transpose(), upperTriangular = false).transpose()
    val C = l \ A / lt
    val CS = C.transpose()
    CS += C
    CS *= 0.5
    val (v, d) = DensePCA.eigenPSDValues(CS, k)
    val V = lt \ v
    // nonzero indices
//    val toTake = d.diag.activeElements.toList.filter(t => t._2 >= threshold).map(_._1)
//    val batch = toTake.size

    val res = GEVResults(V.cols().toList.reverse.map(_.toArray), d.diag.toArray.toList.reverse)

    println(s"ai: $ai bi: $bi cols: ${res.vectors.size} values: ${res.values.size}")

    res

//    val Array(vectors, values) = org.jblas.Eigen.symmetricGeneralizedEigenvectors(A, B)
//    val vecList = vectors.rowsAsList().asScala.toList.map(_.toArray).toList
//    val valueList = values.toArray.toList
//    GEVResults(vecList.take(k), valueList.take(k))
  }

  // 20 16 10 10 250
  def findGEVsBlas(Abdt: BlasDenseTensor2, Binbdt: BlasDenseTensor2, k: Int, gamma: Double): GEVResults = {
    val A = BlasHelpers.transposedDoubleMatrix(Abdt)
    val Bin = BlasHelpers.transposedDoubleMatrix(Binbdt)
    val avgEig = Bin.diag().norm1() / Bin.rows
    val B = Bin.dup()
    val ridge = DoubleMatrix.eye(Bin.rows)
    ridge.muli(avgEig * gamma)
    B.addi(ridge)
    val Array(vectors, values) = org.jblas.Eigen.symmetricGeneralizedEigenvectors(A, B)
    val vecList = vectors.rowsAsList().asScala.toList.map(_.toArray).toList
    val valueList = values.toArray.toList
    GEVResults(vecList.reverse.take(k), valueList.reverse.take(k))
  }
}

object GEVTest {
  def main(args: Array[String]): Unit = {

  }
}