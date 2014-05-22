package md

import cc.factorie.app.classify.backend.{Predictor, OptimizablePredictor}
import cc.factorie.la._
import cc.factorie.optimize.{OptimizableObjectives, GradientStep, Example, OptimizableObjective}
import cc.factorie.model.{WeightsMap, WeightsSet, Weights2, Weights, Parameters}
import cc.factorie.util.DoubleAccumulator
import scala.collection.mutable.ArrayBuffer
import md.BlasDenseTensor2

object MLP {

  // should be scalar input, scalar output, scalar gradient,
  // or vector input, vector output, matrix gradient
  trait Activation {
    // TODO how to make this efficient but also not allow inconsistent implementations like "value" features one method not matching "value" features the other?
    def value(prediction: Double): Double
    def gradient(prediction: Double): Double
    def apply(prediction: Double): Double = value(prediction)
  }

  // # of neurons features each hidden layer

  object OptimizableActivations {
    val logistic = new LogisticActivation
    val tanh = new TanhActivation
  }

  class TanhActivation extends Activation {
     // TODO how to make this efficient but also not allow inconsistent implementations like "value" features one method not matching "value" features the other?
     override def value(prediction: Double): Double = 2.0 * math.log(1.0 / (1 + math.exp(-prediction))) - 1.0
     override def gradient(prediction: Double): Double = {
       val probCorrect = 1.0 / (1 + math.exp(-prediction))
      2.0 * (1 - probCorrect) - 1.0
     }
   }
  class LogisticActivation extends Activation {
    // TODO how to make this efficient but also not allow inconsistent implementations like "value" features one method not matching "value" features the other?
    override def value(prediction: Double): Double = math.log(1.0 / (1 + math.exp(-prediction)))
    override def gradient(prediction: Double): Double = {
      val probCorrect = 1.0 / (1 + math.exp(-prediction))
      1 - probCorrect
    }
  }

  // NOTE this returns transpose gradient G^T (|input| rows, |output| columns
  // NOTE this makes everything add up to 1, so it should only be used for final layer
  class SoftmaxActivation(val T: Double = 1.0) {
    // TODO how to make this efficient but also not allow inconsistent implementations like "value" features one method not matching "value" features the other?
    def value(prediction: DenseTensor1): DenseTensor1 = {val p = prediction.copy; p.expNormalize(); p}
    def gradient(prediction: DenseTensor1): DenseTensor2 = {
      // jacobian matrix: inputs are columns, outputs are rows
      // TODO use transpose so we left multiply? yes!
      // gradients wrt i'th input involve i'th element of value vector
      val predArr = prediction.asArray
      val value = prediction.copy
      if (T != 1.0) value *= T
      val Z = value.expNormalize()
      val valArr = value.asArray
      val d = value.size
      val grad = new DenseTensor2(d, d)
      val gradArr = grad.asArray
      var i = 0
      while (i < d) {
        var j = 0
        while (j < d) {
          // remember this is transpose gradient
          gradArr(i * d + j) = (math.exp(T * predArr(i)) / Z) * (1 + T * (predArr(i) - valArr(j)))
          j += 1
        }
        i += 1
      }
      grad
    }
  }


  // LinearInteraction could be name for vector-vector predictor that uses matrix mult
  // WeightedGradientExample - does this make sense? what about models that need to include some weights while calculating other gradients? features that case it's really part of the model?
  // Also, what about adding different hyperparameters to different templates/Weights features a model? Need a generic dictionary OptimizerOptions class then, with DI/serialization
  // and maybe a generic OptimizerJournal too, like Breeze. Would be awesome to be able to automate runs by specifying an OptimizerOptions by cmd line args or something
  // have CmdOption of type OptimizerOptions with automatic parser --experiment1-optimizer-options="l1: 0.001; l2: 0.00001" json or whatever
  // OptimizerOption: DoubleOption, BooleanOption, IntOption
  // OptimizerOptions: Map[String, OptimizerOption] w/ nice getters
  // RunOptions: Map[Type, Any] ... or should it be Map[Type, RunOption] with some predefined serializable things? could pass around domains like this either way which would be nice
  // would be nice to use those type-level string things as keys K for value type V
  // def resolveService[K](implicit kd: KeyDef[K, V]): V
  // no, easier than this
  // def resolveService[V](kd: KeyDef[V]): V
  // just key off singleton objects

  // this sounds just like CmdOptions then? Or i guess not since we could have multiple runs controlled from one cmdline thing.
  // one problem is multiple constructors - how do we deal with that? just make factory methods that take the container (implicitly even)



  // How would DI work? associated with some marker type? type class approach to pull services out of some container?
  // could I extend the type class DI trick to let you specify a marker type? that might be too many levels of redirection for syntax to work

  // this is kind of annoying to make efficient - features order to reweight the gradient we need to make a new WeightsMap
  // so should this be features the optimizer? Sure, it could be a gradientStep w/ processGradient
  // is this not just per-template learning rates?
  // not if it depended on the current values I guess

//  class WeightedGradientExample(val baseExample: Example, val templateWeights: Map[Weights, Double], val reweightValues: Boolean = true) extends Example {
//    override def accumulateValueAndGradient(value: DoubleAccumulator, gradient: WeightsMapAccumulator): Unit = gradient.bl
//  }

  trait Weighted extends GradientStep {
    val templateWeights: Map[Weights, Double]
    override def processGradient(weights: WeightsSet, gradient: WeightsMap): Unit = {
      super.processGradient(weights, gradient)
      for (k <- gradient.keys) {
        gradient(k) *= templateWeights(k)
      }
    }
  }

  // would be nice to make this be a real predictor - but don't want to accumulate big crazy tensor
  // you'd get one for each example
  // do we want to have a list of accumulators?
  // do we want the option to accumulate the gradient all features one go?
  // what about optimizers that can take advantage?
  // currently optimizers get gradients features sparse or dense tensors, not slices of matrices
  // so for the moment, maybe we just do batch



  trait BatchLayer[Prediction <: Tensor2, Input <: Tensor2] extends OptimizablePredictor[Prediction, Input] {
    val weights: Weights2
    def inputSize: Int
    def outputSize: Int
    def weightsValue: BlasDenseTensor2 = weights.value.asInstanceOf[BlasDenseTensor2]
    def predictFromScores(scores: Tensor2): Prediction
    def predict(input: Input): Prediction = predictFromScores(scores(input))
    def scores(input: Input): Tensor2 =
      input.asInstanceOf[BlasDenseTensor2] * weightsValue
    def calculateObjectiveGradientByScore(scores: Tensor2, objectiveByPredictionGradient: Prediction, scratchPad: Tensor2 = null): Tensor2
    def accumulateObjectiveGradientFromScores(accumulator: WeightsMapAccumulator, input: Input, scores: Tensor2, objectiveByPredictionGradient: Prediction, weight: Double): Unit = {
      val objByScore = calculateObjectiveGradientByScore(scores, objectiveByPredictionGradient)
      accumulator.accumulate(weights, objByScore.asInstanceOf[BlasDenseTensor2] leftMultiply input, weight)
    }
    override def accumulateObjectiveGradient(accumulator: WeightsMapAccumulator, input: Input, objectiveByPredictionGradient: Prediction, weight: Double): Unit =
      accumulateObjectiveGradientFromScores(accumulator, input, scores(input), objectiveByPredictionGradient, weight)
  }

  trait BatchLinearLayer[Prediction <: Tensor2, Input <: Tensor2] extends BatchLayer[Prediction, Input] {
    override def calculateObjectiveGradientByScore(scores: Tensor2, objectiveByPredictionGradient: Prediction, scratchPad: Tensor2 = null): Tensor2 = objectiveByPredictionGradient
    override def predictFromScores(scores: Tensor2): Prediction = scores.copy.asInstanceOf[Prediction]
  }

  // add bias?
  trait BatchNonlinearLayer[Prediction <: Tensor2, Input <: Tensor2] extends BatchLayer[Prediction, Input] {
     def activationFunction: Activation

     override def predictFromScores(scores: Tensor2): Prediction = {
       val res = scores.copy.asInstanceOf[Prediction]
       TensorHelpers.applyFunction(res, activationFunction.value)
       res
     }
     def calculateObjectiveGradientByScore(scores: Tensor2, objectiveByPredictionGradient: Prediction, scratchPad: Tensor2 = null): Tensor2 = {
       val grad = scores.copy
       TensorHelpers.applyFunction(grad, activationFunction.gradient)
 //        val (predValue, predictionByScoresGradient) = activationFunction.valueAndGradient(scores)
       grad *= objectiveByPredictionGradient // TODO make helper for elementwise/hadamard?
       grad
     }
   }




//  trait BatchOptimizablePredictor[Prediction, Input] extends Predictor[Prediction, Input] {
//    def calculateObjectiveGradients(input: Input, objectiveByPredictionGradient: Prediction, weight: Array[Double]): Seq[WeightsMap] = {
//
//    }
//  }
//
//  trait BatchLayer[Prediction <: Tensor2, Input <: Tensor2] extends OptimizablePredictor[Prediction, Input] {
//    val weights: Weights2
//    def inputSize: Int
//    def outputSize: Int
//    def scores(input: Input): Tensor2 = weights.value leftMultiply input
//    def calculateObjectiveGradientByScore(input: Input, objectiveByPredictionGradient: Prediction, scratchPad: Tensor2 = null): Tensor2
//    override def accumulateObjectiveGradient(accumulator: WeightsMapAccumulator, input: Input, objectiveByPredictionGradient: Prediction, weight: Double): Unit =
//      accumulator.accumulate(weights, input outer calculateObjectiveGradientByScore(input, objectiveByPredictionGradient), weight)
//  }


  trait Layer[Prediction <: Tensor1, Input <: Tensor1] extends OptimizablePredictor[Prediction, Input] {
    val weights: Weights2
    def inputSize: Int
    def outputSize: Int
    def predictFromScores(scores: Tensor1): Prediction
    def predict(input: Input): Prediction = predictFromScores(scores(input))
    def scores(input: Input): Tensor1 = weights.value leftMultiply input
    def calculateObjectiveGradientByScore(scores: Tensor1, objectiveByPredictionGradient: Prediction, scratchPad: Tensor1 = null): Tensor1
    override def accumulateObjectiveGradient(accumulator: WeightsMapAccumulator, input: Input, objectiveByPredictionGradient: Prediction, weight: Double): Unit =
      accumulator.accumulate(weights, input outer calculateObjectiveGradientByScore(scores(input), objectiveByPredictionGradient), weight)
  }

  trait LinearLayer[Prediction <: Tensor1, Input <: Tensor1] extends Layer[Prediction, Input] {
    // todo make := be efficient for Outer2Tensors copying into DenseTensors
    override def calculateObjectiveGradientByScore(scores: Tensor1, objectiveByPredictionGradient: Prediction, scratchPad: Tensor1 = null): Tensor1 = objectiveByPredictionGradient
    override def predictFromScores(scores: Tensor1): Prediction = scores.copy.asInstanceOf[Prediction]
  }

  trait NonlinearLayer[Prediction <: Tensor1, Input <: Tensor1] extends Layer[Prediction, Input] {
    def activationFunction: Activation

    override def predictFromScores(scores: Tensor1): Prediction = {
      val res = scores.copy.asInstanceOf[Prediction]
      TensorHelpers.applyFunction(res, activationFunction.value)
      res
    }

    def calculateObjectiveGradientByScore(scores: Tensor1, objectiveByPredictionGradient: Prediction, scratchPad: Tensor1 = null): Tensor1 = {

      // scores by weights gradient is a 3-tensor - weights features each row don't effect scores for other rows
      // scoresByWeightsgradient = just feature vector outer product identity matrix? yes
      // so dPred/dWeights = dPred/dScores dScores/dWeights
      // so what do we do here? presumably this is dPred/dScores outer input
      // dObjective/dWeights = dObjective/dPrediction dPred/dScores dScores/dWeights
      // so this is dObj/dPred vector leftmultiplied into dPred/dScores outer'd with input


      // nice, covariance of gradients is just by chain rule
      // orthogonal xforms dont translate gradients and regular vectors differently
      // presumably since they are orthogonal matrices so Q^T = Q^-1

      val objectiveByScoresGradient = scores.copy
      TensorHelpers.applyFunction(objectiveByScoresGradient, activationFunction.gradient)
//        val (predValue, predictionByScoresGradient) = activationFunction.valueAndGradient(scores)
      objectiveByScoresGradient *= objectiveByPredictionGradient // TODO make helper for elementwise/hadamard?
      objectiveByScoresGradient
    }
  }

  class DenseBatchNonlinearLayer(val inputSize: Int, val outputSize: Int, val activationFunction: Activation = OptimizableActivations.tanh, override val parameters: WeightsSet = new WeightsSet)
    extends BatchNonlinearLayer[BlasDenseTensor2, BlasDenseTensor2] with Parameters {
    val weights = Weights(new BlasDenseTensor2(inputSize, outputSize))
  }

  class DenseBatchLinearLayer(val inputSize: Int, val outputSize: Int, override val parameters: WeightsSet = new WeightsSet) extends BatchLinearLayer[BlasDenseTensor2, BlasDenseTensor2] with Parameters {
    val weights = Weights(new BlasDenseTensor2(inputSize, outputSize))
  }

  class DenseNonlinearLayer(val inputSize: Int, val outputSize: Int, val activationFunction: Activation = OptimizableActivations.tanh, override val parameters: WeightsSet = new WeightsSet)
    extends NonlinearLayer[DenseTensor1, DenseTensor1] with Parameters {
    val weights = Weights(new BlasDenseTensor2(inputSize, outputSize))
  }

  class BatchLayerExample[Output, ModelOut <: Tensor2, ObjectiveIn, Input <: Tensor2](model: BatchLayer[ModelOut, Input], input: Input, label: Output, objective: OptimizableObjective[ObjectiveIn, Output], weight: Double = 1.0)
    extends Example {
    def accumulateValueAndGradient(value: DoubleAccumulator, gradient: WeightsMapAccumulator) {
      val scores = model.scores(input)
      val prediction = model.predictFromScores(scores).asInstanceOf[ObjectiveIn]
      val (obj, ograd) = objective.valueAndGradient(prediction, label)
      if (value != null) value.accumulate(obj * weight)
      if (gradient != null) model.accumulateObjectiveGradientFromScores(gradient, input, scores, ograd.asInstanceOf[ModelOut], weight)
    }
  }

  class MyPredictorExample[Output, ModelOut, ObjectiveIn, Input](model: OptimizablePredictor[ModelOut, Input], input: Input, label: Output, objective: OptimizableObjective[ObjectiveIn, Output], weight: Double = 1.0)
    extends Example {
    def accumulateValueAndGradient(value: DoubleAccumulator, gradient: WeightsMapAccumulator) {
      val prediction = model.predict(input).asInstanceOf[ObjectiveIn]
      val (obj, ograd) = objective.valueAndGradient(prediction, label)
      if (value != null) value.accumulate(obj * weight)
      if (gradient != null) model.accumulateObjectiveGradient(gradient, input, ograd.asInstanceOf[ModelOut], weight)
    }
  }

  class DenseLinearLayer(val inputSize: Int, val outputSize: Int, override val parameters: WeightsSet = new WeightsSet) extends LinearLayer[DenseTensor1, DenseTensor1] with Parameters {
    val weights = Weights(new BlasDenseTensor2(inputSize, outputSize))
  }

  // TODO nice to be able to train a CRF on top of this...
  // label could be seq of one hots?
  // input and output and label could be TensorSet? or just specifically one-hots?

  class DenseMultilayerPerceptron(val inputSize: Int, val hiddenSize: Int, val predictionSize: Int, val k: Int, val activation: Activation = OptimizableActivations.tanh)
    extends OptimizablePredictor[Tensor1, DenseTensor1] with Parameters {

    val layers: Seq[Layer[DenseTensor1, DenseTensor1] with Parameters] = {
      val unitCounts = inputSize +: (0 until k).map(_ => hiddenSize) :+ predictionSize
      val initial = unitCounts.sliding(2).toVector.dropRight(1).map(w => new DenseNonlinearLayer(w(0), w(1), parameters = parameters))
      val finalSizes = unitCounts.takeRight(2)
      initial :+ new DenseLinearLayer(finalSizes(0), finalSizes(1), parameters = parameters)
    }
    val predictionFunction = layers.map(l => l.predict(_)).reduceLeft(_ andThen _)

    def initializeRandomly(stdDev: Double = 0.01)(implicit random: scala.util.Random): Unit =
      layers.map(_.weights.value.asInstanceOf[DenseTensor2]).foreach(dt => {BlasHelpers.randnfill(dt.asArray); dt *= stdDev})

    override def predict(input: DenseTensor1): DenseTensor1 = predictionFunction(input)
    override def accumulateObjectiveGradient(accumulator: WeightsMapAccumulator, input: DenseTensor1, objectiveByPredictionGradient: Tensor1, weight: Double): Unit = {
      val inputs = new collection.mutable.HashMap[Parameters, Tensor1]
      val scores = new collection.mutable.HashMap[Parameters, Tensor1]
      var curInput = input
      for (l <- layers) {
        val curScore = l.scores(curInput)
        scores(l) = curScore
        inputs(l) = curInput
        curInput = l.predictFromScores(curScore)
      }

      val gradients = new collection.mutable.HashMap[Parameters, Tensor1]
      var curObjGradient = objectiveByPredictionGradient // since last layer is linear this is obj gradient by score
      for (l <- layers.reverse) {
        gradients(l) = curObjGradient
        curObjGradient = l.weights.value * l.calculateObjectiveGradientByScore(scores(l), curObjGradient.asInstanceOf[DenseTensor1])
      }

      for (l <- layers)
        accumulator.accumulate(l.weights, inputs(l) outer gradients(l), weight)
    }
  }

  class DenseBatchMultilayerPerceptron(val inputSize: Int, val hiddenSize: Int, val predictionSize: Int, val k: Int, val activation: Activation = OptimizableActivations.tanh)
    extends OptimizablePredictor[BlasDenseTensor2, BlasDenseTensor2] with Parameters {

    val layers: Seq[BatchLayer[BlasDenseTensor2, BlasDenseTensor2]] = {
      val unitCounts = inputSize +: (0 until k).map(_ => hiddenSize) :+ predictionSize
      val initial = unitCounts.sliding(2).toVector.dropRight(1).map(w => new DenseBatchNonlinearLayer(w(0), w(1), parameters = parameters))
      val finalSizes = unitCounts.takeRight(2)
      initial :+ new DenseBatchLinearLayer(finalSizes(0), finalSizes(1), parameters = parameters)
    }
    val predictionFunction = layers.map(l => l.predict(_)).reduceLeft(_ andThen _)

    def initializeRandomly(stdDev: Double = 0.01)(implicit random: scala.util.Random): Unit =
      layers.map(_.weights.value.asInstanceOf[DenseTensor2]).foreach(dt => {BlasHelpers.randnfill(dt.asArray); dt *= stdDev})

    override def predict(input: BlasDenseTensor2): BlasDenseTensor2 = predictionFunction(input)
    override def accumulateObjectiveGradient(accumulator: WeightsMapAccumulator, input: BlasDenseTensor2, objectiveByPredictionGradient: BlasDenseTensor2, weight: Double): Unit = {
      val inputs = layers.scanLeft(input)((value, l) => l.predict(value)).dropRight(1)
      val inputMap = layers.zip(inputs).toMap
      val objGradientsByScore = layers.scanRight(objectiveByPredictionGradient)((l, gradient) =>
        (l.weightsValue * l.calculateObjectiveGradientByScore(inputMap(l), gradient)).asInstanceOf[BlasDenseTensor2]).drop(1)
      val objGradientsMap = layers.zip(objGradientsByScore).toMap

      for (l <- layers)
        accumulator.accumulate(l.weights, inputMap(l) outer objGradientsMap(l), weight)
    }
  }

}
