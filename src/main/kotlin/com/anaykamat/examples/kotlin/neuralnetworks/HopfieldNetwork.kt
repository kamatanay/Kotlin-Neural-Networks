package com.anaykamat.examples.kotlin.neuralnetworks

data class HopfieldNetwork private constructor(val weights:List<Int>, val inputSize:Int) {


    private val weightMatrix:Either<MatrixError, Matrix<Int>> = matrixBuilder(inputSize)(inputSize)(weights)

    fun think(pattern:List<Int>):List<Int> =
        matrixBuilder(1)(pattern.size)(pattern).flatMap { patternMatrix ->
            weightMatrix.flatMap {
                it.multiply(patternMatrix.transpose())
            }
        }.fold(emptyList<Int>(),{_, matrix -> matrix.items}).map {
            when{
                it > 0 -> 1
                else -> -1
            }
        }



    companion object {

        private fun rotateRight(list:List<Int>):List<Int> = listOf(list.last())+list.subList(0,list.size-1)

        private fun <I> ((I) -> I).andThenRepeat(times:Int):(I) -> I = { input ->
            ((0 until times).map{this}).fold(input, {output, f -> f(output)})
        }

        private val matrixBuilder =
            (Matrix.builder<Int>()::from).toCurried()({ x, y -> x + y })({ x, y -> x - y })({ x, y -> x * y })


        fun forInputSize(inputSize: Int):HopfieldNetwork = HopfieldNetwork((0 until inputSize*inputSize).map { 0 }, inputSize)

        fun train(network:HopfieldNetwork, inputs:List<Int>):HopfieldNetwork{
            return network.weightMatrix.flatMap { currentWeightMatrix ->
                generateWeightMatrix(inputs).flatMap { weightMatrix ->
                    identityMatrix(inputs.size).flatMap { identityMatrix ->
                        weightMatrix.subtract(identityMatrix)
                    }.flatMap { newWeightMatrix -> newWeightMatrix.add(currentWeightMatrix) }
                }
            }.fold(network,{_, matrix -> HopfieldNetwork(matrix.items, network.inputSize)})
        }

        private fun generateWeightMatrix(inputs:List<Int>):Either<MatrixError, Matrix<Int>>{
            val inputColumnMatrix = matrixBuilder(inputs.size)(1)(inputs)
            return inputColumnMatrix.flatMap { inputColumn ->
                inputColumn.multiply(inputColumn.transpose())
            }
        }

        private fun identityMatrix(inputSize:Int):Either<MatrixError, Matrix<Int>>{
            val firstRow = (listOf(1)+(0 until inputSize - 1).map { 0 })
            val identityValues = ((0 until inputSize).map { Triple(this::rotateRight, it, firstRow) })
                .fold(emptyList<Int>(),{output, processingData -> processingData.run { output + first.andThenRepeat(second)(third) }})
            return matrixBuilder(inputSize)(inputSize)(identityValues)
        }

    }

}