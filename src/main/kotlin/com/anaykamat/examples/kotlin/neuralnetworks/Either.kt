package com.anaykamat.examples.kotlin.neuralnetworks

class ForEither private constructor(){
    companion object
}

fun <A,B> Kind<Kind<ForEither,A>,B>.fix():Either<A,B> = this as Either<A,B>


fun <L> ForEither.Companion.monad():Monad<Kind<ForEither,L>> = object:Monad<Kind<ForEither,L>>{
    override fun <A> pure(value: A): Kind<Kind<ForEither, L>, A> = Either.Right(value)

    override fun <A, B> Kind<Kind<ForEither, L>, A>.map(f: (A) -> B): Kind<Kind<ForEither, L>, B> = this.fix().map(f)

    override fun <A, B> Kind<Kind<ForEither, L>, A>.flatMap(f: (A) -> Kind<Kind<ForEither, L>, B>): Kind<Kind<ForEither, L>, B> = this.fix().flatMap(f)

}

sealed class Either<out A, out B>:Kind<Kind<ForEither, A>,B>{

    data class Right<out B>(val data:B):Either<Nothing,B>()
    data class Left<out A>(val data:A):Either<A,Nothing>()

    fun <C> map(f:(B) -> C):Either<A,C> = when(this){
        is Right -> Either.Right(f(this.data))
        is Left -> this
    }

    fun take(count: Int): Kind<A,List<B>> = when(this){
        is Right -> Right((0 until count).map { this.data }) as Kind<A, List<B>>
        is Left -> Left(this.data) as Kind<A,List<B>>
    }

    fun <I> fold(initial:I, f:(I,B) -> I):I = when(this){
        is Right -> f(initial, this.data)
        is Left -> initial
    }

}

infix fun <A,B,C> Either<A,B>.flatMap(f:(B) -> Either<A,C>):Either<A,C> = when(this){
    is Either.Right -> f(this.data)
    is Either.Left -> this
}