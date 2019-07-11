package com.anaykamat.examples.kotlin.neuralnetworks

import java.util.concurrent.atomic.AtomicReference
import kotlin.coroutines.*
import kotlin.coroutines.intrinsics.suspendCoroutineUninterceptedOrReturn
import kotlin.coroutines.intrinsics.COROUTINE_SUSPENDED
import kotlin.coroutines.CoroutineContext

class MonadContinuation<F,A>(private val monad:Monad<F>):Continuation<Kind<F,A>>, Monad<F> by monad {

    val value:AtomicReference<Kind<F,A>> = AtomicReference()

    override val context: CoroutineContext
        get() = EmptyCoroutineContext

    override fun resumeWith(result: Result<Kind<F, A>>) {
        result.onSuccess {
            value.set(it)
        }
    }

    suspend fun <B> Kind<F,B>.bind():B = bind({this})

    suspend fun <B> bind(f:()-> Kind<F,B>):B{
        return suspendCoroutineUninterceptedOrReturn { continuation ->
            (f().flatMap { it ->
                continuation.resume(it)
                value.get()
            }).let { value.set(it) }
            COROUTINE_SUSPENDED
        }
    }


}

fun <F,A> Monad<F>.binding(block:suspend MonadContinuation<F,*>.() -> A):Kind<F,A>{

    val wrapReturn:suspend MonadContinuation<F,*>.() -> Kind<F,A> = {
        pure(
            block()
        )
    }
    val receiver = MonadContinuation<F,A>(this)
    wrapReturn.startCoroutine(receiver, receiver)
    return receiver.value.get()
}