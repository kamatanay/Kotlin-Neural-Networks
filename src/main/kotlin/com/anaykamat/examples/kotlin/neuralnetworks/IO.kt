package com.anaykamat.examples.kotlin.neuralnetworks

class ForIO private constructor(){
    companion object
}

fun <A> Kind<ForIO,A>.fix():IO<A> = this as IO<A>


fun ForIO.Companion.monad():Monad<ForIO> = object:Monad<ForIO>{
    override fun <A> pure(value: A): Kind<ForIO, A> = IO {value}

    override fun <A, B> Kind<ForIO, A>.map(f: (A) -> B): Kind<ForIO, B> = this.fix().map(f)

    override fun <A, B> Kind<ForIO, A>.flatMap(f: (A) -> Kind<ForIO, B>): IO<B> = this.fix().flatMap(f as (A) -> IO<B>)
}

class IO<A>(val runUnsafe:() -> A):Kind<ForIO,A> {

    fun <B> map(f:(A) -> B):IO<B> = IO {f(runUnsafe())}
    fun <B> flatMap(f:(A) -> IO<B>):IO<B> = IO {f(runUnsafe()).runUnsafe()}
    fun take(count: Int): IO<List<A>> = IO {(0..count-1).map { runUnsafe() }}

}