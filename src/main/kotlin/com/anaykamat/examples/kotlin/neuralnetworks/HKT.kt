package com.anaykamat.examples.kotlin.neuralnetworks

interface Kind<out F, out A>

interface Monad<F>{
    fun <A> pure(value:A): Kind<F,A>
    fun <A,B> Kind<F,A>.map(f:(A) -> B): Kind<F,B>
    fun <A,B> Kind<F,A>.flatMap(f:(A) -> Kind<F,B>): Kind<F,B>
}
