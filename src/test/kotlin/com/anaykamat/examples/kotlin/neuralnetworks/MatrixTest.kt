package com.anaykamat.examples.kotlin.neuralnetworks

import org.junit.Assert
import org.junit.Test

class MatrixTest {

    val integerMatrixBuilder = Matrix.builder<Int>()::from.toCurried()({x,y -> x+y})({x,y -> x-y})({x,y -> x*y})

    @Test
    fun itShouldCreateAMatrixForGivenRowsAndColumnsAndWithRequiredOperationsForSpecifiedType(){
        val matrix = integerMatrixBuilder(2)(2)(listOf(1,2,3,4))
        matrix.map {
            Assert.assertTrue(it::class.java == Matrix::class.java)
        }
    }

    @Test
    fun itShouldReturnErrorIfTheItemsInInputListDoNotMatchSizeOfMatrix(){
        val matrix = integerMatrixBuilder(2)(2)(listOf(1,2,3,4,5))
        Assert.assertEquals(Either.Left(MatrixError.InputSizeNotMatchingWithMatrixSize), matrix)
    }

    @Test
    fun transposeShouldGenerateTheTransposeOfTheGivenMatrix(){
        val matrix = integerMatrixBuilder(3)(3)(listOf(1,2,3,4,5,6,7,8,9))
        val expectedMatrix = integerMatrixBuilder(3)(3)(listOf(1,4,7,2,5,8,3,6,9))
        matrix.map {
            val transposedMatrix = it.transpose()
            Assert.assertEquals(expectedMatrix, Either.Right(transposedMatrix))
        }
    }

    @Test
    fun dotShouldFindTheDotProductOfItems(){
        val result = ForEither.monad<MatrixError>().binding {
            val firstMatrix = integerMatrixBuilder(1)(2)(listOf(1,2)).bind()
            val secondMatrix = integerMatrixBuilder(1)(2)(listOf(3,4)).bind()

            firstMatrix.dot(secondMatrix).bind()
        }

        Assert.assertEquals(integerMatrixBuilder(1)(1)(listOf(11)), result)
    }

    @Test
    fun dotShouldReturnErrorIfTheMatrixIsNotOfSameDimension(){
        val firstMatrix = integerMatrixBuilder(1)(2)(listOf(1,2))
        val secondMatrix = integerMatrixBuilder(1)(3)(listOf(3,4,5))

        val result = firstMatrix.flatMap { firstMatrix ->
            secondMatrix.flatMap { secondMatrix ->
                firstMatrix.dot(secondMatrix)
            }
        }

        Assert.assertEquals(Either.Left(MatrixError.DimensionsAreNotEqual), result)
    }

    @Test
    fun chunkedShouldGiveMatrixDividedIntoGroupsOfGivenDimensions(){
        val matrix = integerMatrixBuilder(4)(4)((1..16).toList())

        val result = matrix.flatMap{ it.chunked(2,2) }

        val matrixBuilder2By2 = integerMatrixBuilder(2)(2)

        val expectedValues = ForEither.monad<MatrixError>().binding {
            val matrix1 = matrixBuilder2By2(listOf(1,2,5,6)).bind()
            val matrix2 = matrixBuilder2By2(listOf(3,4,7,8)).bind()
            val matrix3 = matrixBuilder2By2(listOf(9,10,13,14)).bind()
            val matrix4 = matrixBuilder2By2(listOf(11,12,15,16)).bind()
            listOf(matrix1, matrix2, matrix3, matrix4)
        }

        Assert.assertEquals(expectedValues, result)
    }

    @Test
    fun chunkedShouldGiveErrorIfMatrixCannotBeDividedIntoRequiredChunks(){
        val matrix = integerMatrixBuilder(4)(4)((1..16).toList())

        val result = matrix.flatMap{ it.chunked(3,2) }


        Assert.assertEquals(Either.Left(MatrixError.IncompatibleMatrixDimensions), result)
    }

    @Test
    fun multiplyShouldMultiplyTheTwoGivenMatrices(){
        val firstMatrix = integerMatrixBuilder(3)(3)((1..9).toList())
        val secondMatrix = integerMatrixBuilder(3)(2)((1..6).toList())

        val result = firstMatrix.flatMap { first ->
            secondMatrix.flatMap { first.multiply(it) }
        }

        Assert.assertEquals(integerMatrixBuilder(3)(2)(listOf(22, 28, 49, 64, 76, 100)), result)
    }

    @Test
    fun multiplyShouldGiveErrorIfTheColumnsIfFirstMatrixIsNotEqualToRowsOfSecondMatrix(){
        val firstMatrix = integerMatrixBuilder(3)(3)((1..9).toList())
        val secondMatrix = integerMatrixBuilder(6)(2)((1..12).toList())

        val result = firstMatrix.flatMap { first ->
            secondMatrix.flatMap { first.multiply(it) }
        }

        Assert.assertEquals(Either.Left(MatrixError.IncompatibleMatrixDimensions), result)
    }

    @Test
    fun addShouldAddValuesOfFirstMatrixWithEveryCorrespondingValueOfSecondMatrix(){
        val firstMatrix = integerMatrixBuilder(3)(3)((1..9).toList())
        val secondMatrix = integerMatrixBuilder(3)(3)((1..9).toList())

        val expectedMatrix = integerMatrixBuilder(3)(3)((1..9).map { it * 2 })

        val result = firstMatrix.flatMap { first ->
            secondMatrix.flatMap { first.add(it) }
        }

        Assert.assertEquals(expectedMatrix, result)
    }

    @Test
    fun addShouldGivenErrorIfDimensionsAreNotSame(){
        val firstMatrix = integerMatrixBuilder(3)(3)((1..9).toList())
        val secondMatrix = integerMatrixBuilder(1)(1)((1..1).toList())


        val result = firstMatrix.flatMap { first ->
            secondMatrix.flatMap { first.add(it) }
        }

        Assert.assertEquals(Either.Left(MatrixError.IncompatibleMatrixDimensions), result)
    }

    @Test
    fun subtractShouldSubtractValuesOfSecondMatrixFromEveryCorrespondingValueOfFirstMatrix(){
        val firstMatrix = integerMatrixBuilder(3)(3)((2..10).toList())
        val secondMatrix = integerMatrixBuilder(3)(3)((1..9).toList())

        val expectedMatrix = integerMatrixBuilder(3)(3)((1..9).map { 1 })

        val result = firstMatrix.flatMap { first ->
            secondMatrix.flatMap { first.subtract(it) }
        }

        Assert.assertEquals(expectedMatrix, result)
    }

    @Test
    fun subtractShouldGivenErrorIfDimensionsAreNotSame(){
        val firstMatrix = integerMatrixBuilder(3)(3)((1..9).toList())
        val secondMatrix = integerMatrixBuilder(1)(1)((1..1).toList())


        val result = firstMatrix.flatMap { first ->
            secondMatrix.flatMap { first.subtract(it) }
        }

        Assert.assertEquals(Either.Left(MatrixError.IncompatibleMatrixDimensions), result)
    }

}