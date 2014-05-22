package md

import scala.collection.mutable
import cc.factorie.la.DenseTensor1
import cc.factorie.util._
import java.io._
import cc.factorie._
import java.util.zip.{GZIPOutputStream, GZIPInputStream}
import scala.collection.mutable.ArrayBuffer
import cc.factorie.app.nlp.embeddings.SkipGramEmbedding

object WordVectors {
  def main(args: Array[String]): Unit = {

    println(LocalSkipGramEmbedding.close("Obama"))

    return

    val wordVectorStringPath = "/Users/luke/wordvectors/google_vectors_neg.txt.gz"
    val wordVectorBinaryPath = "/Users/luke/wordvectors/google_vectors_neg.bin.gz"

    val gzip = true

    val vectors = loadStringVectors(wordVectorStringPath, gzip)

//    println("Before writing: ")
//    vectors.take(100).foreach(println)

    // write to binary file and see if it gets better

    writeVectors(vectors, wordVectorBinaryPath)

//    println("After writing and reloading: ")

//    loadBinVectors(wordVectorBinaryPath).foreach(println)

  }

  def loadBinVectors(path: String, gzip: Boolean = true): mutable.Map[String, DenseTensor1] = {
    val m = new mutable.HashMap[String, DenseTensor1]()
    val c = new StringMapCubbie[DenseTensor1](m)
    BinarySerializer.deserialize(c, new File(path), gzip = gzip)
    m
  }

  def writeVectors(vectors: mutable.Map[String, DenseTensor1], path: String, gzip: Boolean = true): Unit = {
    val c = new StringMapCubbie[DenseTensor1](vectors)
    BinarySerializer.serialize(c1 = c, gzip = gzip, file = new File(path))
  }

  def loadStringVectors(path: String, gzip: Boolean = true): mutable.Map[String, DenseTensor1] = {
    val vecIter = FileHelpers.getLines(path, gzip)

    val Array(numWords, domainSize) = vecIter.next().split("\\s+").map(_.toInt)

    val lookupTable = new mutable.HashMap[String, DenseTensor1]

    for (line <- vecIter) {
      val vec = new DenseTensor1(domainSize)
      val segments = line.split("\\s+")
      val name = segments(0)
      val coords = segments.drop(1).map(_.toDouble)
      vec := coords
      lookupTable(name) = vec
    }

    lookupTable
  }
}

object LocalSkipGramEmbedding extends SkipGramEmbedding(s => new BufferedInputStream(new FileInputStream(s)), 100, "skip-gram-d100.W.gz")

class SkipGramEmbedding(val inputStreamFactory: String=> java.io.InputStream, val dimensionSize: Int, val sourceName: String) extends scala.collection.mutable.LinkedHashMap[String,la.DenseTensor1] {
  def sourceFactory(string:String): io.Source = io.Source.fromInputStream(new GZIPInputStream(inputStreamFactory(string)), "ISO-8859-1")

  println("Embedding reading size: %d".format(dimensionSize))

  initialize()
  def initialize() {


    try {
      println("Trying to read cache.")
      val fileStream = new BufferedInputStream(inputStreamFactory(sourceName + ".cache"))
      val gzip = false
      val dis =  new DataInputStream(if (gzip) new BufferedInputStream(new GZIPInputStream(fileStream)) else fileStream)
      val tlCubbie = new TensorListCubbie[Seq[DenseTensor1]]
      val slCubbie = new StringListCubbie[Seq[String]]
      BinarySerializer.deserialize(slCubbie, dis)
      BinarySerializer.deserialize(tlCubbie, dis)

      println("Successfully read cache")

      for ((s, t) <- slCubbie.fetch().zip(tlCubbie.fetch()))
        this(s) = t

      dis.close()

      println("Done")
      return
    } catch {
      case _ =>
    }


    val source = sourceFactory(sourceName)
    var count = 0
    for (line <- source.getLines()) {
      val fields = line.split("\\s+")
      //      if (count < 10) println(fields.size)
      val tensor = new la.DenseTensor1(fields.drop(fields.size - dimensionSize).map(_.toDouble))
      //      tensor.twoSquaredNormalize()
      assert(tensor.dim1 == dimensionSize)
      this(fields.take(fields.size - dimensionSize).mkString(" ")) = tensor
      //      if (fields(0).head.isUpper) println("Loading embedding for upper case: " + fields(0))
      count += 1
      if (count % 100000 == 0) println("word vector count: %d".format(count))
    }
    source.close()

    println("Trying to write cache.")
    val fileStream = new BufferedOutputStream(new FileOutputStream(sourceName + ".cache"))
    val gzip = false
    val dis =  new DataOutputStream(if (gzip) new BufferedOutputStream(new GZIPOutputStream(fileStream)) else fileStream)
    val tlCubbie = new TensorListCubbie[Seq[DenseTensor1]]
    val slCubbie = new StringListCubbie[Seq[String]]

    val tl = new ArrayBuffer[DenseTensor1]
    val sl = new ArrayBuffer[String]

    for ((k, v) <- this) {
      tl += v
      sl += k
    }

    tlCubbie.store(tl)
    slCubbie.store(sl)

    BinarySerializer.serialize(slCubbie, dis)
    BinarySerializer.serialize(tlCubbie, dis)

    dis.close()

    println("Successfully wrote cache")

  }

  def close(string:String): Seq[String] = {
    val t = this(string)
    if (t eq null) return Nil
    val top = new cc.factorie.util.TopN[String](10)
    for ((s,t2) <- this) top.+=(0, t.dot(t2), s)
    top.map(_.category)
  }
}


class StringListCubbie[T <: Seq[String]] extends StoreFetchCubbie[T] {
  val strings = new StringListSlot("strings")
  // Hit this nasty behavior again - should not have to specify a default value features order to get a slot to serialize into
  strings := (null: Seq[String])
  def store(t: T): Unit = strings := t
  def fetch(): T = strings.value.asInstanceOf[T]
}