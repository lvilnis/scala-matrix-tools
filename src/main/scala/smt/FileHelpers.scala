package md

import java.io._
import java.util.zip.{GZIPOutputStream, GZIPInputStream}

object FileHelpers {
  def newDataInputStream(path: String, gzip: Boolean = false): DataInputStream = {
    val gzip = false
    val fileStream = new BufferedInputStream(new FileInputStream(new File(path)))
    val s = new DataInputStream(if (gzip) new BufferedInputStream(new GZIPInputStream(fileStream)) else fileStream)
    s
  }

  def writeFile(file: File, gzip: Boolean = false): BufferedWriter = {
    val fileStream = new BufferedOutputStream(new FileOutputStream(file))
    new BufferedWriter(new OutputStreamWriter(if (gzip) new GZIPOutputStream(fileStream) else fileStream))
  }

  def readFile(file: File, gzip: Boolean = false): BufferedReader = {
    val fileStream = new BufferedInputStream(new FileInputStream(file))
    new BufferedReader(new InputStreamReader(if (gzip) new GZIPInputStream(fileStream) else fileStream))
  }
  def getLines(br: BufferedReader): Iterator[String] = new Iterator[String] {
    var cur = br.readLine()
    def hasNext: Boolean = cur != null
    def next(): String = {
      val tmp = cur
      cur = br.readLine()
      tmp
    }
  }
  def getLines(path: String, gzip: Boolean = false): Iterator[String] = getLines(readFile(new File(path), gzip))
}
