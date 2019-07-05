package com.anaykamat.examples.kotlin.neuralnetworks

import org.junit.Assert
import org.junit.Test

class IOMonadTest {

    @Test
    fun itShouldSupportFlatMap(){
        val runUnsafe = { 2 }
        val expectedIOMonad = IO(runUnsafe)

        val flatMappedValue = ForIO.monad().pure(10).fix().flatMap { value -> if (value == 10) expectedIOMonad else IO {value} }

        val result = flatMappedValue.map {
            Assert.assertEquals(2, it)
        }

        result.runUnsafe()
    }

    @Test
    fun itShouldAllowUserToGetIOWithListOfMonadValues(){
        val runUnsafe = { 2 }
        val ioMonad = IO(runUnsafe)

        val monad:IO<List<Int>> = ioMonad.take(5)

        val result = monad.map {
            Assert.assertEquals(listOf(2,2,2,2,2), it)
        }

        result.runUnsafe()
    }


}