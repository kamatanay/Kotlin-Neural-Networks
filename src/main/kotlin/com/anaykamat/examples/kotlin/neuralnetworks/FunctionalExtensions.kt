package com.anaykamat.examples.kotlin.neuralnetworks

fun <A,B,C,D,E,F,G> ((A,B,C,D,E,F) -> G).toCurried():(A) -> (B) -> (C) -> (D) -> (E) -> (F) -> G {
    return { a ->
        { b ->
            { c ->
                {d ->
                    {e ->
                        { f->
                            this(a,b,c,d,e,f)
                        }
                    }

                }
            }

        }

    }
}