package com.anaykamat.examples.kotlin.neuralnetworks

interface Activation<T> {
    fun activationFor(value:T):T
    fun gradientOf(value:T):T
}