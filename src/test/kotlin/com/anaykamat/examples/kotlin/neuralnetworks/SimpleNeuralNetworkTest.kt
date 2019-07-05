package com.anaykamat.examples.kotlin.neuralnetworks

import org.junit.Assert
import org.junit.Test
import kotlin.random.Random

class SimpleNeuralNetworkTest {

    @Test
    fun thinkShouldPredictTheOutputValueBasedOnNeuronWeightsAndInputs(){

        val activation:Activation<Double> = object:Activation<Double>{
            override fun activationFor(value: Double): Double  = value * 0.2

            override fun gradientOf(value: Double): Double = value / 0.2

        }

        val neuralNetwork = SimpleNeuralNetwork(listOf(0.1,0.2,0.3), 3, activation)
        val result:List<Double> = neuralNetwork.think(listOf(listOf(0.0,1.0,1.0)))

        Assert.assertEquals(listOf(0.1), result)
    }

    @Test
    fun trainShouldGiveNeuralNetworkWithAdjustedWeight(){

        val activation:Activation<Double> = object:Activation<Double>{
            override fun activationFor(value: Double): Double  = value * 0.2

            override fun gradientOf(value: Double): Double = value / 0.2

        }

        val neuralNetwork = SimpleNeuralNetwork(listOf(0.1,0.2,0.3), 3, activation)

        val trainedNeuralNetwork = SimpleNeuralNetwork.train(neuralNetwork, listOf(listOf(0.0,0.2,0.0)), listOf(listOf(0.0)), 1)


        listOf(0.1,0.199936,0.3).zip(trainedNeuralNetwork.weights) { expected, actual ->
            Assert.assertEquals(expected, actual, 0.0000009)
        }


    }

    @Test
    fun trainShouldRepeatTheProcessForTheGivenCount(){

        val activation:Activation<Double> = object:Activation<Double>{
            override fun activationFor(value: Double): Double  = value * 0.2

            override fun gradientOf(value: Double): Double = value / 0.2

        }

        val neuralNetwork = SimpleNeuralNetwork(listOf(0.1,0.2,0.3), 3, activation)

        val trainedNeuralNetwork = SimpleNeuralNetwork.train(neuralNetwork, listOf(listOf(0.0,0.2,0.0)), listOf(listOf(0.0)), 2)


        listOf(0.1,0.199872,0.3).zip(trainedNeuralNetwork.weights) { expected, actual ->
            Assert.assertEquals(expected, actual, 0.0000009)
        }


    }

    @Test
    fun actualTest(){
        val neuralNetworkProgram = Random(0).let { IO { it.nextDouble(-1.0, 1.0) } }.take(3).map { randomWeights ->
            val neuralNetwork = SimpleNeuralNetwork(randomWeights, 3, SigmoidActivation())
            val trainedNeuralNetwork = SimpleNeuralNetwork.train(neuralNetwork, listOf(
                0, 0, 0,
                0, 0, 1,
                1, 0, 1,
                1, 1, 0,
                0, 1, 1
            ).map { it.toDouble() }.chunked(3), listOf(0, 0, 1, 1, 0).map { it.toDouble() }.chunked(1), 30_000)
            trainedNeuralNetwork.think(listOf(1, 0, 0).map { it.toDouble() }.chunked(3))
        }

        val value = neuralNetworkProgram.runUnsafe()
        Assert.assertEquals(1.0, value.first(),0.09)

    }


}