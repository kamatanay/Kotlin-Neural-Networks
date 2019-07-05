package com.anaykamat.examples.kotlin.neuralnetworks

import org.junit.Assert
import org.junit.Test

class EitherMonadType {
    @Test
    fun itShouldRunTheFlatMapOperationIfValueIsOfTypeRight(){
        val rightValue = Either.Right<Int>(10)
        val flatMappedResult = rightValue.flatMap {
            Either.Right(true)
        }
        Assert.assertEquals(Either.Right(true), flatMappedResult)
    }

    @Test
    fun itShouldNotRunTheFlatMapOperationIfValueIsOfTypeLeft(){
        val leftValue = Either.Left("There was an error")
        val flatMappedResult = leftValue.flatMap {
            Either.Right(true)
        }
        Assert.assertEquals(Either.Left("There was an error"), flatMappedResult)
    }

    @Test
    fun itShouldRunTheMapOperationIfValueIsOfTypeRight(){
        val rightValue = Either.Right<Int>(10)
        val flatMappedResult = rightValue.map {
            true
        }
        Assert.assertEquals(Either.Right(true), flatMappedResult)
    }

    @Test
    fun itShouldNotRunTheMapOperationIfValueIsOfTypeLeft(){
        val leftValue = Either.Left("There was an error")
        val flatMappedResult = leftValue.map {
            Either.Right(true)
        }
        Assert.assertEquals(Either.Left("There was an error"), flatMappedResult)
    }

    @Test
    fun takeShouldReturnTheListOfValuesIfTheTypeIsRight(){
        val rightValue = Either.Right<Int>(10)
        val take5 = rightValue.take(5)
        Assert.assertEquals(Either.Right(listOf(10,10,10,10,10)), take5)
    }

    @Test
    fun takeShouldTheLeftValueIfTypeIsLeft(){
        val leftValue = Either.Left("There was an error")
        val take5 = leftValue.take(5)
        Assert.assertEquals(Either.Left("There was an error"), take5)
    }

}