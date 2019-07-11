package com.anaykamat.examples.kotlin.neuralnetworks

data class SimpleNeuralNetwork(val weights:List<Double>, val layerCount:Int, val activation:Activation<Double>) {


    private val weightMatrix:Either<MatrixError, Matrix<Double>> = matrixBuilder(layerCount)(1)(weights)

    fun think(inputs:List<List<Double>>):List<Double> = ForEither.monad<MatrixError>().binding {
        matrixBuilder(inputs.size)(layerCount)(inputs.flatten())
            .bind()
            .multiply(weightMatrix.bind())
            .bind()
            .map(activation::activationFor)
    }.fix().fold(listOf(0.0),{ _, resultMatrix -> resultMatrix.items})


    companion object {

        val matrixBuilder =
            (Matrix.builder<Double>()::from).toCurried()({ x, y -> x + y })({ x, y -> x - y })({ x, y -> x * y })

        tailrec fun train(neuralNetwork: SimpleNeuralNetwork, trainingInputs: List<List<Double>>, expectedOutputs: List<List<Double>>, trainingSteps: Int): SimpleNeuralNetwork {

            val updatedNetwork = ForEither.monad<MatrixError>().binding {
                val output = neuralNetwork.think(trainingInputs)
                val outputMatrix = matrixBuilder(output.size)(1)(output).bind()
                val expectedOutputMatrix = matrixBuilder(output.size)(1)(expectedOutputs.flatten()).bind()

                val inputMatrix = matrixBuilder(trainingInputs.size)(3)(trainingInputs.flatten()).bind()

                val errorMatrix = expectedOutputMatrix.subtract(outputMatrix)
                    .bind()
                    .apply({x,y -> x*y}, outputMatrix.map(neuralNetwork.activation::gradientOf))

                val weightChangeDelta = inputMatrix.transpose().multiply(errorMatrix).bind()

                matrixBuilder(neuralNetwork.layerCount)(1)(neuralNetwork.weights)
                    .bind()
                    .add(weightChangeDelta)
                    .bind()
                    .let {
                        neuralNetwork.copy(weights = it.items)
                    }
            }.fix().fold(neuralNetwork) { _, modifiedNeuralNetwork -> modifiedNeuralNetwork}

            return when{
                trainingSteps < 1 -> neuralNetwork
                trainingSteps == 1 -> updatedNetwork
                else -> train(updatedNetwork, trainingInputs, expectedOutputs, trainingSteps - 1)
            }

        }

    }

}