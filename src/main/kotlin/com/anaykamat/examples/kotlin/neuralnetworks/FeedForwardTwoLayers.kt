package com.anaykamat.examples.kotlin.neuralnetworks

data class FeedForwardTwoLayers(val inputLayerWeights:List<Double>, val hiddenLayerWeights:List<Double>, val inputCount:Int, val hiddenNeuronCount:Int, val activation:Activation<Double>) {

    val inputWeightMatrix = matrixBuilder(inputCount)(hiddenNeuronCount)(inputLayerWeights)
    val hiddenWeightMatrix = matrixBuilder(hiddenNeuronCount)(1)(hiddenLayerWeights)

    fun think(inputs:List<List<Double>>):List<Double> =
        inputMatrix(inputs, inputCount)
            .flatMap { inputs -> calculateLayerOutput(inputs, inputWeightMatrix, activation) }
            .flatMap { hiddenLayerInput -> calculateLayerOutput(hiddenLayerInput, hiddenWeightMatrix, activation) }
            .fold(emptyList(),{ _, outputMatrix -> outputMatrix.items})

    companion object {
        val matrixBuilder = Matrix.builder<Double>()::from.toCurried()({x,y -> x+y})({x,y -> x-y})({x,y -> x*y})

        tailrec fun learn(network:FeedForwardTwoLayers, testInputs:List<List<Double>>, testOutput:List<Double>, iterations:Int):FeedForwardTwoLayers{
            val inputMatrix = inputMatrix(testInputs, network.inputCount)

            val hiddenLayerData = inputMatrix.flatMap { calculateLayerOutput(it, network.inputWeightMatrix, network.activation) }
            val output = hiddenLayerData.flatMap { calculateLayerOutput(it, network.hiddenWeightMatrix,network.activation) }

            val expectedOutputMatrix = matrixBuilder(testInputs.size)(1)(testOutput)

            val outputErrorMatrix = output.flatMap { output ->
                expectedOutputMatrix.flatMap { expected -> expected.subtract(output) }
                    .map { error -> error.apply({ x, y -> x * y }, output.map(network.activation::activationFor)) }
            }

            val hiddenError = outputErrorMatrix.flatMap { outputError ->
                hiddenLayerData.flatMap { hiddenData ->
                    network.hiddenWeightMatrix.flatMap { hiddenWeights ->
                        outputError.multiply(hiddenWeights.transpose()).map { it.apply({x,y -> x*y}, hiddenData.map(network.activation::gradientOf)) }
                    }
                }
            }

            val deltaOutputMatrix = hiddenLayerData.flatMap { hiddenData -> outputErrorMatrix.flatMap { outputError -> hiddenData.transpose().multiply(outputError) } }
            val deltaHiddenMatrix = inputMatrix.flatMap { inputData -> hiddenError.flatMap { hiddenErrorData -> inputData.transpose().multiply(hiddenErrorData) } }

            val newInputWeights = deltaHiddenMatrix.flatMap { deltaWeights -> network.inputWeightMatrix.flatMap { existingWeights -> deltaWeights.add(existingWeights) } }
            val newOutputWeights = deltaOutputMatrix.flatMap { deltaWeights -> network.hiddenWeightMatrix.flatMap { existingWeights -> deltaWeights.add(existingWeights) } }

            val newNetwork = newInputWeights.flatMap { inputWeights ->
                newOutputWeights.map { outputWeights ->
                    network.copy(inputLayerWeights = inputWeights.items, hiddenLayerWeights = outputWeights.items)
                }
            }.fold(network, {_, network -> network})

            return when(iterations > 0){
                true -> learn(newNetwork, testInputs, testOutput, iterations-1)
                false -> newNetwork
            }
        }

        private fun inputMatrix(
            inputs: List<List<Double>>,
            inputCount: Int
        ) = matrixBuilder(inputs.size)(inputCount)(inputs.flatten())

        private fun calculateLayerOutput(inputMatrix:Matrix<Double>, weightMatrix:Either<MatrixError, Matrix<Double>>, activation:Activation<Double>):Either<MatrixError, Matrix<Double>> =
                weightMatrix.flatMap { weights -> inputMatrix.multiply(weights) }.map { it.map(activation::activationFor) } }

}
