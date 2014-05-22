package md

import java.io._
import java.util.zip.GZIPInputStream
import cc.factorie.variable.{CategoricalVectorDomain, CategoricalDomain}
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import cc.factorie._
import cc.factorie.la._
import cc.factorie.optimize._
import cc.factorie.app.classify.backend.{OptimizablePredictor, LinearMulticlassClassifier, MulticlassClassifier}
import cc.factorie.model.{Weights2, WeightsMap}
import scala.io.Source

object TacRelationDomain extends CategoricalDomain[String] {
  this ++= Vector(
    "org:member_of",
    "per:city_of_death",
    "per:country_of_birth",
    "per:stateorprovince_of_death",
    "org:country_of_headquarters",
    "per:schools_attended",
    "org:political_religious_affiliation",
    "per:countries_of_residence",
    "per:stateorprovince_of_birth",
    "org:members",
    "per:title",
    "org:subsidiaries",
    "per:charges",
    "per:employee_or_member_of",
    "per:parents",
    "org:parents",
    "org:shareholders",
    "per:country_of_death",
    "per:children",
    "per:city_of_birth",
    "org:founded_by",
    "org:website",
    "per:cause_of_death",
    "per:origin",
    "per:spouse",
    "org:number_of_employees_members",
    "per:statesorprovinces_of_residence",
    "per:cities_of_residence",
    "org:alternate_names",
    "org:city_of_headquarters",
    "per:date_of_death",
    "per:religion",
    "org:stateorprovince_of_headquarters",
    "per:other_family",
    "org:top_members_employees",
    "per:alternate_names",
    "org:date_founded",
    "org:date_dissolved",
    "per:siblings",
    "per:date_of_birth",
    "per:age")
  freeze()
}

object OptimizableObjectiveHelper {
  def batchMulticlassObjective(obj: OptimizableObjectives.Multiclass): BatchMulticlassOptimizableObjective = {
    ???
  }
  def multiclassOneVsAll(obj: OptimizableObjectives.Binary, numClasses: Int): OptimizableObjectives.Multiclass = new MultivariateOptimizableObjective[Int] {
    def valueAndGradient(prediction: Tensor1, label: Int): (Double, Tensor1) = {
      val gradient = new DenseTensor1(numClasses)
      var value = 0.0
      var i = 0
      while (i < numClasses) {
        val (v, g) = obj.valueAndGradient(prediction(i), if (label == i) 1 else -1)
        value += v
        gradient(i) = g
        i += 1
      }
      (value, gradient)
    }
  }
}

object Relations {
  val relationListPath = """/Users/luke/tac/relationfactory/config/relations2013.config"""

  val distantSupervisionPath = """/Users/luke/tac/data/merge_2013.tab.gz"""
  val keyDataPath = """/Users/luke/tac/data/gold_candidates"""
  val candidatesPath = """/Users/luke/tac/data/candidates"""

  val predictionOutputPath = """/Users/luke/tac/data/predictions"""

  case class RelationInstance(
    arg1: String, relation: String, arg2: String, id: String, arg1Start: Int,
    arg1End: Int, arg2Start: Int, arg2End: Int, sentence: Seq[String], entityPair: String,
    rawTextLine: String)

  case class ProcessedRelationInstance(relation: String, label: Int, feats: ProcessedRelationFeatures)
  case class ProcessedRelationFeatures(replacedTokens: Seq[String], arg1First: Boolean, arg1Index: Int, arg2Index: Int, entityPair: String, rawTextLine: String)

  case class FeaturizedRelationInstance(feats: GrowableSparseIndexedTensor1, label: Int, entityPair: String)

  def processRelationInstance(in: RelationInstance): ProcessedRelationInstance = {
    val arg1First = in.arg1Start < in.arg2Start
    val replaced =
      if (arg1First) replaceTokensWithOne(replaceTokensWithOne(in.sentence, in.arg2Start, in.arg2End, "ARG2"), in.arg1Start, in.arg1End, "ARG1")
      else replaceTokensWithOne(replaceTokensWithOne(in.sentence, in.arg1Start, in.arg1End, "ARG1"), in.arg2Start, in.arg2End, "ARG2")
    val res = ProcessedRelationInstance(in.relation, TacRelationDomain.index(in.relation),
      ProcessedRelationFeatures(replaced, arg1First, replaced.indexOf("ARG1"), replaced.indexOf("ARG2"), in.entityPair, in.rawTextLine))
    //    println(res)
    res
  }

  // put left or right of query marker into template name
  def addNGrams(tokens: Seq[String], start: Int, end: Int, templateName: String, k: Int, buffer: ArrayBuffer[String]): Unit = {
    if (start < 0 || end > tokens.size || end - start < k) return
    val realStart = math.min(tokens.size, math.max(0, start))
    val realEnd = math.min(tokens.size, math.max(0, end))
    val relevant = tokens.drop(realStart).take(realEnd - realStart)
    relevant.sliding(k).map(arr => templateName + "#" + arr.mkString("#")).foreach(buffer += _)
  }

  def getNGrams(tokens: Seq[String], start: Int, end: Int, k: Int): Seq[Seq[String]] = {
    if (start < 0 || end > tokens.size || end - start < k) return Seq()
    val realStart = math.min(tokens.size, math.max(0, start))
    val realEnd = math.min(tokens.size, math.max(0, end))
    val relevant = tokens.drop(realStart).take(realEnd - realStart)
    relevant.sliding(k).toSeq.filter(_.size == k)
  }

  def addSkipNGrams(tokens: Seq[String], start: Int, end: Int, templateName: String, k: Int, buffer: ArrayBuffer[String]): Unit = {
    if (start < 0 || end > tokens.size || end - start < k) return
    val realStart = math.min(tokens.size, math.max(0, start))
    val realEnd = math.min(tokens.size, math.max(0, end))
    val relevant = tokens.drop(realStart).take(realEnd - realStart)
    relevant.sliding(k).map(arr => templateName + "#" + arr.head + arr.drop(2).map(_ => "#").mkString + "#" + arr.last).foreach(buffer += _)
  }

  def featurizeRelationInstance(in: ProcessedRelationInstance, featureDomain: CategoricalVectorDomain[String], labelDomain: CategoricalDomain[String]): FeaturizedRelationInstance = {
    val features = in.feats
    val feats = new GrowableSparseIndexedTensor1(featureDomain.dimensionDomain)
    val featsBuffer = new ArrayBuffer[String]
    val startIdx = if (features.arg1First) features.arg1Index else features.arg2Index
    val endIdx = if (features.arg1First) features.arg2Index else features.arg1Index

    addNGrams(features.replacedTokens, start = startIdx + 1, end = endIdx, templateName = "BETWEEN_NGRAM" + (if (features.arg1First) ">" else "<"), k = 1, featsBuffer)
    addNGrams(features.replacedTokens, start = startIdx, end = endIdx + 1, templateName = "BETWEEN_NGRAM" + (if (features.arg1First) ">" else "<"), k = 2, featsBuffer)
    addNGrams(features.replacedTokens, start = startIdx, end = endIdx + 1, templateName = "BETWEEN_NGRAM" + (if (features.arg1First) ">" else "<"), k = 3, featsBuffer)

    addNGrams(features.replacedTokens, start = startIdx - 1, end = startIdx + 1, templateName = "OUTSIDE_NGRAM" + (if (features.arg1First) ">" else "<"), k = 2, featsBuffer)
    addNGrams(features.replacedTokens, start = startIdx - 2, end = startIdx + 1, templateName = "OUTSIDE_NGRAM" + (if (features.arg1First) ">" else "<"), k = 3, featsBuffer)

    addNGrams(features.replacedTokens, start = endIdx, end = endIdx + 2, templateName = "OUTSIDE_NGRAM" + (if (features.arg1First) ">" else "<"), k = 2, featsBuffer)
    addNGrams(features.replacedTokens, start = endIdx, end = endIdx + 3, templateName = "OUTSIDE_NGRAM" + (if (features.arg1First) ">" else "<"), k = 3, featsBuffer)

    addSkipNGrams(features.replacedTokens, start = startIdx, end = endIdx + 1, templateName = "SKIP_NGRAM" + (if (features.arg1First) ">" else "<"), k = 3, featsBuffer)
    addSkipNGrams(features.replacedTokens, start = startIdx, end = endIdx + 1, templateName = "SKIP_NGRAM" + (if (features.arg1First) ">" else "<"), k = 4, featsBuffer)


    featsBuffer.foreach(f => {
      val idx = featureDomain.dimensionDomain.index(f)
      if (idx >= 0) feats +=(idx, 1.0)
    })
    //    println(featsBuffer)
    val res = FeaturizedRelationInstance(feats, labelDomain.index(in.relation), features.entityPair)
    //    println(res)
    res
    //    ???
  }

  def replaceTokensWithOne(tokens: Seq[String], start: Int, end: Int, replacementString: String): Seq[String] = {
    tokens.take(start) ++ Seq(replacementString) ++ tokens.drop(end)
  }

  def groupInstancesAndNormalize(instances: Seq[FeaturizedRelationInstance], featureDomain: CategoricalVectorDomain[String]): Seq[FeaturizedRelationInstance] = {
    val result = new ArrayBuffer[FeaturizedRelationInstance]
    for ((pair, ins) <- instances.groupBy(in => in.entityPair + in.label)) {
      val feats = new GrowableSparseIndexedTensor1(featureDomain.dimensionDomain)
      ins.foreach(in => feats += in.feats)
      val max = feats.max
      feats /= max
      result += FeaturizedRelationInstance(feats, ins.head.label, ins.head.entityPair)
    }
    result
  }

  object CategoricalRelationFeatureDomain extends CategoricalVectorDomain[String] {
    dimensionDomain.gatherCounts = true
  }

  object EmbeddingFeatureDomain extends DiscreteDomain(LocalSkipGramEmbedding.dimensionSize)

  object EmbeddingRelationModelFeatureHelper {
    def featurizeInstances(features: Seq[ProcessedRelationFeatures], model: EmbeddingRelationModel): DenseDesignMatrix = {
      val templateSizes = model.parameters.toSeq.map(_._2.asInstanceOf[Tensor2].dim1).toVector
      val numFeats = templateSizes.sum
      val offsets = model.parameters.keys.zip(templateSizes.scanLeft(0)(_ + _).dropRight(1)).toMap
      val numRelations = TacRelationDomain.dimensionSize
      val mat = new BlasDenseTensor2(features.size, numFeats)
      for ((f , i)<- features.zipWithIndex) {
        val featMap = featurizeInstance(f, model)
        for ((k, v) <- featMap) {
          val offset = offsets(k)
          val varr = v.asArray
          val len = EmbeddingFeatureDomain.dimensionSize
          var j = 0
          while (j < len) {
            mat(numFeats * i + offset + j) = varr(j)
            j += 1
          }
        }
      }
      DenseDesignMatrix(mat, transposed = false)
    }
    def featurizeInstance(features: ProcessedRelationFeatures, model: EmbeddingRelationModel): mutable.Map[Weights2, DenseTensor1] = {
      import model._
      val result = new mutable.HashMap[Weights2, DenseTensor1] {
        override def default(w: Weights2): DenseTensor1 = {
          new DenseTensor1(EmbeddingFeatureDomain.dimensionSize)
        }
      }
      val startIdx = if (features.arg1First) features.arg1Index else features.arg2Index
      val endIdx = if (features.arg1First) features.arg2Index else features.arg1Index

      def addVec(key: Weights2, toAdd: DenseTensor1): Unit = {
        val cur = result(key)
        cur += toAdd
        result(key) = cur
      }

      def multVec(key: Weights2, toMult: Double): Unit = {
        val cur = result(key)
        cur *= toMult
        result(key) = cur
      }

      // TODO just split this into two models?
      if (features.arg1First) {

        if (startIdx - 2 >= 0)
          LocalSkipGramEmbedding.get(features.replacedTokens(startIdx - 2)).foreach(g => addVec(f1fm2, g))
        if (startIdx - 1 >= 0)
          LocalSkipGramEmbedding.get(features.replacedTokens(startIdx - 1)).foreach(g => addVec(f1fm1, g))

        if (endIdx + 2 < features.replacedTokens.size)
          LocalSkipGramEmbedding.get(features.replacedTokens(endIdx + 2)).foreach(g => addVec(f1sp2, g))
        if (endIdx + 1 < features.replacedTokens.size)
          LocalSkipGramEmbedding.get(features.replacedTokens(endIdx + 1)).foreach(g => addVec(f1sp1, g))

        var grams1Count = 0
        val grams1 = getNGrams(features.replacedTokens, start = startIdx + 1, end = endIdx, k = 1)
        grams1.foreach(grams => {
          LocalSkipGramEmbedding.get(grams(0)).foreach(g => {
            addVec(f1between1gram, g)
            grams1Count += 1
          })
        })
        if (grams1Count > 0) {
          multVec(f1between1gram, 1.0 / grams1Count)
        }

        var grams2Count = 0
        val grams2 = getNGrams(features.replacedTokens, start = startIdx + 1, end = endIdx, k = 2)
        grams2.foreach(grams => {
          LocalSkipGramEmbedding.get(grams(0)).foreach(g => {
            addVec(f1between2gram1, g)
            grams2Count += 1
          })
          LocalSkipGramEmbedding.get(grams(1)).foreach(g => {
            addVec(f1between2gram2, g)
            grams2Count += 1
          })
        })
        if (grams2Count > 0) {
          multVec(f1between2gram1, 0.5 / grams2Count)
          multVec(f1between2gram2, 0.5 / grams2Count)
        }
      } else {

        if (startIdx - 2 >= 0)
          LocalSkipGramEmbedding.get(features.replacedTokens(startIdx - 2)).foreach(g => {
            addVec(f2fm2, g)
          })
        if (startIdx - 1 >= 0)
          LocalSkipGramEmbedding.get(features.replacedTokens(startIdx - 1)).foreach(g => addVec(f2fm1, g))

        if (endIdx + 2 < features.replacedTokens.size)
          LocalSkipGramEmbedding.get(features.replacedTokens(endIdx + 2)).foreach(g => addVec(f2sp2, g))
        if (endIdx + 1 < features.replacedTokens.size)
          LocalSkipGramEmbedding.get(features.replacedTokens(endIdx + 1)).foreach(g => addVec(f2sp1, g))

        var grams1Count = 0
        val grams1 = getNGrams(features.replacedTokens, start = startIdx + 1, end = endIdx, k = 1)
        grams1.foreach(grams => {
          LocalSkipGramEmbedding.get(grams(0)).foreach(g => {
            addVec(f2between1gram, g)
            grams1Count += 1
          })
        })
        if (grams1Count > 0) {
          multVec(f2between1gram, 1.0 / grams1Count)
        }

        var grams2Count = 0
        val grams2 = getNGrams(features.replacedTokens, start = startIdx + 1, end = endIdx, k = 2)
        grams2.foreach(grams => {
          LocalSkipGramEmbedding.get(grams(0)).foreach(g => {
            addVec(f2between2gram1, g)
            grams2Count += 1
          })
          LocalSkipGramEmbedding.get(grams(1)).foreach(g => {
            addVec(f2between2gram2, g)
            grams2Count += 1
          })
        })

        if (grams2Count > 0) {
          multVec(f2between2gram1, 0.5 / grams2Count)
          multVec(f2between2gram2, 0.5 / grams2Count)
        }

      }
      result
    }
  }

  // featurize on the fly for now - too slow?
  class EmbeddingRelationModel extends OptimizablePredictor[Tensor1, ProcessedRelationFeatures] with Parameters {
    def newEmbeddingWeights(): Weights2 = Weights(new DenseTensor2(EmbeddingFeatureDomain.dimensionSize, TacRelationDomain.dimensionSize))

    // arg1First - "f1"
    val f1fm1 = newEmbeddingWeights()
    val f1fm2 = newEmbeddingWeights()

    val f1sp1 = newEmbeddingWeights()
    val f1sp2 = newEmbeddingWeights()

    val f1between1gram = newEmbeddingWeights()

    val f1between2gram1 = newEmbeddingWeights()
    val f1between2gram2 = newEmbeddingWeights()

    // arg2First - "f2"
    val f2fm1 = newEmbeddingWeights()
    val f2fm2 = newEmbeddingWeights()

    val f2sp1 = newEmbeddingWeights()
    val f2sp2 = newEmbeddingWeights()

    val f2between1gram = newEmbeddingWeights()

    val f2between2gram1 = newEmbeddingWeights()
    val f2between2gram2 = newEmbeddingWeights()

    // Features


    override def predict(input: ProcessedRelationFeatures): Tensor1 = {
      val feats = EmbeddingRelationModelFeatureHelper.featurizeInstance(input, this)
      val result = new DenseTensor1(TacRelationDomain.dimensionSize)
      feats.keys.foreach(k => {
        result += parameters(k).asInstanceOf[Tensor2].leftMultiply(feats(k))
      })
      result
    }

    override def accumulateObjectiveGradient(accumulator: WeightsMapAccumulator, input: ProcessedRelationFeatures, gradient: Tensor1, weight: Double): Unit = {
      val feats = EmbeddingRelationModelFeatureHelper.featurizeInstance(input, this)
      feats.keys.foreach(k => {
        accumulator.accumulate(k, feats(k) outer gradient, weight)
      })
    }
  }

  implicit val r = new scala.util.Random(0)

  def main(args: Array[String]): Unit = {

    //    trainCategoricalModel()

    val (m, featMap) = trainEmbeddingModel()

    val featureModel = new EmbeddingRelationModel

    val (testInstances, testMatrix, testLabels) = {
      val ins = loadRelationInstances(candidatesPath, gzip = false).map(processRelationInstance)
      val mat = featMap.transform(EmbeddingRelationModelFeatureHelper.featurizeInstances(ins.map(_.feats), featureModel))
      (ins, mat, ins.map(_.label).toArray)
    }

    val scores = BlasHelpers.indexRows(m.predict(testMatrix.mat), testLabels)

    val outputLines = testInstances.zip(scores).map({case (pri, score) =>
      val outputLine = pri.feats.rawTextLine.split("\t").dropRight(1).mkString("\t") + "\t" + score
      outputLine
    })

//    val outputLines = testInstances.map(pri => {
//      val scores = m.predict(pri.feats)
//      val score = scores(pri.label) // if (pri.label == scores.maxIndex) 1.0 else -1.0
//      val outputLine = pri.feats.rawTextLine.split("\t").dropRight(1).mkString("\t") + "\t" + score
//      outputLine
//    })

    val outputStr = outputLines.mkString("\n")

    val f = new File(predictionOutputPath)

    val writer = FileHelpers.writeFile(f)

    writer.write(outputStr)

    writer.close()

    //    val s = Source.fromFile(predictionOutputPath)


  }

  def trainEmbeddingModel(): (MLP.DenseBatchLinearLayer, FeatureMap[DenseDesignMatrix, DenseDesignMatrix]) = {
    val featureModel = new EmbeddingRelationModel

    val (trainMatrixUnmapped, trainLabels, testMatrixUnmapped, testLabels) = {
      val instances = loadRelationInstances(distantSupervisionPath).map(processRelationInstance)
      val (trainInstances, testInstances) = instances.shuffle.split(0.8)
      val trainMat = EmbeddingRelationModelFeatureHelper.featurizeInstances(trainInstances.map(_.feats), featureModel)
      val testMat = EmbeddingRelationModelFeatureHelper.featurizeInstances(testInstances.map(_.feats), featureModel)
      (trainMat, MulticlassLabels(trainInstances.map(_.label).toArray), testMat, MulticlassLabels(testInstances.map(_.label).toArray))
    }

    val classIndices = (0 until TacRelationDomain.dimensionSize).toSeq
    val allPairs = classIndices.flatMap(i => classIndices.map(j => (i, j))).toSeq
    val disallowed = allPairs.filter({case (c1, c2) => c1 == c2 || TacRelationDomain.dimensionDomain.category(c1).take(3) != TacRelationDomain.dimensionDomain.category(c2).take(3)}).toSet
    println(disallowed.size)
    val featureMapTrainer = GEV(k = 10, n = 20, disallowedPairs = disallowed)
//    val featureMapTrainer = FeatureMapTrainer.concat(IdentityFeatureMap().asInstanceOf[FeatureMapTrainer[DenseDesignMatrix, DenseDesignMatrix, MulticlassLabels]], GEV(k = 10)) // DensePCA(k = 50) + RFF(d = 3000)
//    val featureMapTrainer = IdentityFeatureMap()

    val finalMap = featureMapTrainer.trainFeatureMap(trainMatrixUnmapped, trainLabels)

    val (trainMatrix, testMatrix) = (finalMap.transform(trainMatrixUnmapped), finalMap.transform(testMatrixUnmapped))

    val cls = new MLP.DenseBatchLinearLayer(trainMatrix.domainSize, TacRelationDomain.dimensionSize)

    // replace with smooth hinge loss so we can weight prec-rec differently and still optimize with bfgs
    val logBatchMulticlass = new LogBatchMulticlass

    val examples = Seq(new MLP.BatchLayerExample(cls, trainMatrix.mat, trainLabels.labels.toArray, logBatchMulticlass, weight = 10.0))

    Trainer.batchTrain(cls.parameters, examples, useParallelTrainer = false, evaluate = () => {
      // make AdaGrad params variables so we can tweak it
  //    val startingRate = 0.0001
  //    val delta2 = 1
  //    val opt = new MyPerceptron { baseRate = startingRate / delta2 }   //new MyAdaGrad { rate = startingRate; delta = 1.0 }
  //    var i = 0
  //    Trainer.onlineTrain(cls.parameters, examples, maxIterations = 100, optimizer = opt, useParallelTrainer = false, evaluate = () => {

        println("Train accuracy: ")
  //      val numTrainCorrect = mnistTrainSet.count({case (f, l) => l == cls.predict(f).maxIndex})
  //      println(numTrainCorrect * 1.0 / mnistTrainSet.size)
        println(trainLabels.evaluateAccuracy(cls.predict(trainMatrix.mat)))

        println("Test accuracy: ")
  //      val numTestCorrect = mnistTestSet.count({case (f, l) => l == cls.predict(f).maxIndex})
  //      println(numTestCorrect * 1.0 / mnistTestSet.size)
        println(testLabels.evaluateAccuracy(cls.predict(testMatrix.mat)))

  //      opt.reset()
  //      i += 1
  //      opt.baseRate = startingRate / (delta2 + i)
      })


//    val singleClassObj = OptimizableObjectives.hingeScaledBinary(posSlackRescale = 7.0)
//
//    val objective = OptimizableObjectiveHelper.multiclassOneVsAll(singleClassObj, TacRelationDomain.dimensionSize)
//
//    // write something to turn this into a DenseDesignMatrix so i can use my feature maps...
//
//    val examples = trainInstances.map(in => new PredictorExample(model, in.feats, in.label, objective))
//
//    Trainer.onlineTrain(model.parameters, examples, maxIterations = 1, evaluate = () => {
//      println("Train accuracy: " + evaluate(trainInstances, model))
//      println("Test accuracy: " + evaluate(testInstances, model))
//    })

    (cls, finalMap)
  }

//  def trainCategoricalModel(): Unit = {
//    val featureDomain = CategoricalRelationFeatureDomain
//
//    val instances = {
//      loadRelationInstances(distantSupervisionPath).map(in =>
//        featurizeRelationInstance(processRelationInstance(in), featureDomain, TacRelationDomain))
//
//      println("Domain size before pruning: " + featureDomain.dimensionSize)
//
//      //      featureDomain.dimensionDomain.trimBelowCount(3)
//      featureDomain.freeze()
//
//      println("Domain size after pruning: " + featureDomain.dimensionSize)
//
//      groupInstancesAndNormalize(
//        loadRelationInstances(distantSupervisionPath).map(in =>
//          featurizeRelationInstance(processRelationInstance(in), featureDomain, TacRelationDomain)),
//        featureDomain)
//    }
//
//
//
//    //    val model = new OptimizablePredictor[Int, Tensor1] with Parameters {
//    //      val weights = Weights(new DenseTensor2(featureDomain.dimensionSize, TacRelationDomain.size))
//    //    }
//    //
//
//    class CategoricalRelationModel extends LinearMulticlassClassifier(TacRelationDomain.dimensionSize, EmbeddingFeatureDomain.dimensionSize)
//
//    val model = new CategoricalRelationModel
//
//    val (trainInstances, testInstances) = instances.shuffle.split(0.8)
//
//    val singleClassObj = OptimizableObjectives.hingeScaledBinary(posSlackRescale = 3.0)
//
//    val objective = OptimizableObjectiveHelper.multiclassOneVsAll(singleClassObj, TacRelationDomain.dimensionSize)
//
//    val examples = trainInstances.map(in => new PredictorExample(model, in.feats, in.label, objective))
//
//
//    Trainer.onlineTrain(model.parameters, examples, evaluate = () => {
//      println("Train accuracy: " + evaluate(trainInstances, model))
//      println("Test accuracy: " + evaluate(testInstances, model))
//    })
//
//  }



  def evaluate(ins: Seq[ProcessedRelationInstance], model: EmbeddingRelationModel): Double = {
    ins.count(evaluate(_, model)) * 1.0 / ins.size
  }

  def evaluate(in: ProcessedRelationInstance, model: EmbeddingRelationModel): Boolean = {
    in.label == model.predict(in.feats).maxIndex
  }

  def evaluate(ins: Seq[FeaturizedRelationInstance], model: LinearMulticlassClassifier): Double = {
    ins.count(evaluate(_, model)) * 1.0 / ins.size
  }

  def evaluate(in: FeaturizedRelationInstance, model: LinearMulticlassClassifier): Boolean = {
    in.label == model.predict(in.feats).maxIndex
  }

  // validate on key data?
  //  def loadKeyData(path: String):


  // batch multiclass
  // batch binary
  // batch one-vs-all

  def loadRelationInstances(path: String, gzip: Boolean = true): Seq[RelationInstance] = {
    val vecIter = FileHelpers.getLines(path, gzip = gzip)
    val lines = vecIter.toVector.shuffle.take(100000)
    val result = new ArrayBuffer[RelationInstance]()
    for (line <- lines /*.take(10000)*/ ) {
      val Array(arg1, relation, arg2, id, arg1StartStr, arg1EndStr, arg2StartStr, arg2EndStr, sentenceStr) = line.split("\t")
      val pair = arg1 + arg2
      result += RelationInstance(arg1, relation, arg2, id, arg1StartStr.toInt, arg1EndStr.toInt, arg2StartStr.toInt, arg2EndStr.toInt, sentenceStr.split(" "), pair, line)
    }
    result
  }

  def printRelationList(): Unit = {
    val gzip = false
    val s = FileHelpers.newDataInputStream(relationListPath, gzip)
    val relations = new mutable.HashSet[String]()
    var line = s.readLine()
    for (_ <- 0 until 200) {
      if (line != null) {
        relations += line.split(' ').head
        line = s.readLine()
      }
    }
    relations.toSeq.foreach(println)
  }
}

