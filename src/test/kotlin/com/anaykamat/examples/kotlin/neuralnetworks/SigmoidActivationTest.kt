package com.anaykamat.examples.kotlin.neuralnetworks

import org.junit.Assert
import org.junit.Test

class SigmoidActivationTest {

    @Test
    fun itShouldGiveSigmoidResultForGivenInputValue(){
        val input = 0.2
        ForSigmoid.activation().run{
            Assert.assertEquals(0.5498, activationFor(input),0.00009)
        }
    }

    @Test
    fun itShouldGradientOfGivenValue(){
        val input = 0.5498
        ForSigmoid.activation().run{
            Assert.assertEquals(0.2, gradientOf(input),0.09)
        }
    }

}