package com.anaykamat.examples.kotlin.neuralnetworks

import org.junit.Assert
import org.junit.Test
import kotlin.math.roundToInt
import kotlin.random.Random

class FeedForwardTwoLayersTest {

    @Test
    fun actualTest(){
        val neuralNetworkProgram = Random(0).let { IO { it.nextDouble(-1.0, 1.0) } }.take(9).map { randomWeights ->
            val neuralNetwork = FeedForwardTwoLayers(randomWeights.subList(0,6), randomWeights.subList(6, randomWeights.size), 2, 3, SigmoidActivation())
            val trainedNeuralNetwork = FeedForwardTwoLayers.learn(neuralNetwork, listOf(
                0, 0,
                0, 1,
                1, 0,
                1, 1
            ).map { it.toDouble() }.chunked(2), listOf(0, 1, 1, 0).map { it.toDouble() }, 617)

            trainedNeuralNetwork.think(listOf(
                0, 0,
                0, 1,
                1, 0,
                1, 1
            ).map { it.toDouble() }.chunked(2))
        }

        val value = neuralNetworkProgram.runUnsafe()
        Assert.assertEquals(listOf(0.0,1.0,1.0,0.0), value.map { it.roundToInt().toDouble() })
    }

}