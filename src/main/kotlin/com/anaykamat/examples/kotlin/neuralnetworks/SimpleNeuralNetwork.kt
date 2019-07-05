package com.anaykamat.examples.kotlin.neuralnetworks

data class SimpleNeuralNetwork(val weights:List<Double>, val layerCount:Int, val activation:Activation<Double>) {

    private val weightMatrix:Either<MatrixError, Matrix<Double>> = matrixBuilder(layerCount)(1)(weights)


    fun think(inputs:List<List<Double>>):List<Double> = matrixBuilder(inputs.size)(layerCount)(inputs.flatten())
        .flatMap { inputMatrix -> weightMatrix.flatMap { inputMatrix.multiply(it) } }
        .map { outputMatrix -> outputMatrix.map(activation::activationFor) }
        .fold(listOf(0.0)) { _, resultMatrix -> resultMatrix.items}

    companion object {

        val matrixBuilder =
            (Matrix.builder<Double>()::from).toCurried()({ x, y -> x + y })({ x, y -> x - y })({ x, y -> x * y })

        tailrec fun train(neuralNetwork: SimpleNeuralNetwork, trainingInputs: List<List<Double>>, expectedOutputs: List<List<Double>>, trainingSteps: Int): SimpleNeuralNetwork {
            val output = neuralNetwork.think(trainingInputs)
            val outputMatrix = matrixBuilder(output.size)(1)(output)
            val expectedOutputMatrix = matrixBuilder(output.size)(1)(expectedOutputs.flatten())

            val inputMatrix = matrixBuilder(trainingInputs.size)(3)(trainingInputs.flatten())

            val weightChangeDelta = outputMatrix.flatMap { outputM ->
                expectedOutputMatrix.flatMap { expectedOutputM ->
                    val error = expectedOutputM.subtract(outputM)
                    error.flatMap { errorM ->
                        outputM
                            .map { neuralNetwork.activation.gradientOf(it) }
                            .apply({x,y -> x*y}, errorM)
                            .let { lossM ->
                                inputMatrix.flatMap { it.transpose().multiply(lossM) } }

                    }
                }
            }

            val updatedNetwork = matrixBuilder(neuralNetwork.layerCount)(1)(neuralNetwork.weights).flatMap { weightMatrix ->
                weightChangeDelta.flatMap { weightDeltaMatrix -> weightMatrix.add(weightDeltaMatrix) }
            }.map {
                neuralNetwork.copy(weights = it.items)
            }.fold(neuralNetwork) { _, modifiedNeuralNetwork -> modifiedNeuralNetwork}

            return when{
                trainingSteps < 1 -> neuralNetwork
                trainingSteps == 1 -> updatedNetwork
                else -> train(updatedNetwork, trainingInputs, expectedOutputs, trainingSteps - 1)
            }
        }

    }

}