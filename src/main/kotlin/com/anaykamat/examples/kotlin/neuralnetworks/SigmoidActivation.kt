package com.anaykamat.examples.kotlin.neuralnetworks


class ForSigmoid private constructor() {
    companion object {
        fun activation():Activation<Double> = SigmoidActivation()
    }
}

class SigmoidActivation():Activation<Double> {
    override fun activationFor(value: Double): Double = 1 / (1+Math.exp(-value))

    override fun gradientOf(value: Double): Double = value * (1 - value)
}