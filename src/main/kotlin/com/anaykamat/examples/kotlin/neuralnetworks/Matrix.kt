package com.anaykamat.examples.kotlin.neuralnetworks

sealed class MatrixError{
    object InputSizeNotMatchingWithMatrixSize:MatrixError()
    object DimensionsAreNotEqual:MatrixError()
    object IncompatibleMatrixDimensions:MatrixError()
}

class MatrixBuilder<T>(private val buildMethod:((T, T) -> T,(T, T) -> T, (T,T) -> T,Int,Int,List<T>) -> Either<MatrixError, Matrix<T>>){
    fun from( addFunction:(T, T) -> T,
              subtractFunction: (T,T) -> T,
                  multiplicationFunction: (T,T) -> T,
                  rows:Int,
                  columns:Int,
                  items:List<T>):Either<MatrixError, Matrix<T>>{

        return buildMethod(addFunction, subtractFunction, multiplicationFunction, rows, columns, items)

    }
}

data class Matrix<T> private constructor(val addFunction:(T, T) -> T,
                                         val subtractFunction:(T,T) -> T,
                                         val multiplicationFunction: (T,T) -> T,
                                         val rows:Int,
                                         val columns:Int,
                                         val items:List<T>){
    fun transpose(): Matrix<T> = this.splitIntoChunks(1,rows, items, columns).flatten().let {
        this.copy(rows = columns, columns = rows, items = it)
    }

    fun add(other:Matrix<T>):Either<MatrixError, Matrix<T>> = when(rows == other.rows && columns == other.columns){
        true -> apply(addFunction, other).let { Either.Right(it) }
        else -> Either.Left(MatrixError.IncompatibleMatrixDimensions)
    }

    fun subtract(other:Matrix<T>):Either<MatrixError, Matrix<T>> = when(rows == other.rows && columns == other.columns){
        true -> apply(subtractFunction, other).let { Either.Right(it) }
        else -> Either.Left(MatrixError.IncompatibleMatrixDimensions)
    }


    fun dot(otherMatrix:Matrix<T>):Either<MatrixError, Matrix<T>>{

        return when(listOf(rows, columns) == listOf(otherMatrix.rows, otherMatrix.columns)){
            true -> this.items
                        .zip(otherMatrix.items,{x,y -> multiplicationFunction(x,y)})
                        .reduceRight(addFunction)
                        .let {
                            this.copy(items = listOf(it), rows = 1, columns = 1)
                        }.let {
                            Either.Right(it)
                        }
            false -> Either.Left(MatrixError.DimensionsAreNotEqual)
        }


    }

    fun multiply(other:Matrix<T>):Either<MatrixError, Matrix<T>>{
        return when(columns == other.rows){
            true -> this.chuncked(1, columns).flatMap { rowVectors ->
                        other.chuncked(columns, 1).flatMap { columnVectors ->
                            rowVectors.fold(Either.Right(emptyList<Matrix<T>>()) as Either<MatrixError, List<Matrix<T>>>,{ finalList, rowMatrix ->

                                columnVectors.fold(Either.Right(emptyList<Matrix<T>>()) as Either<MatrixError, List<Matrix<T>>>,{ rowList, columnMatrix ->
                                    rowMatrix.dot(columnMatrix.transpose()).flatMap { value -> rowList.map { it + listOf(value) } }
                                }).flatMap { rowList ->
                                    finalList.map { it + rowList }
                                }

                            })
                        }
                    }
            else -> Either.Left(MatrixError.IncompatibleMatrixDimensions)
        }.map {
            it.map { it.items }.flatten().let { this.copy(items = it, rows = rows, columns = other.columns) }
        }
    }

    fun chuncked(rowCount:Int, columnCount:Int):Either<MatrixError, List<Matrix<T>>>{
        return when(rows % rowCount == 0 && columns % columnCount == 0){
            true -> splitIntoChunks(columnCount, rowCount, items, columns).map {
                            this.copy(items = it, rows = rowCount, columns = columnCount)
                        }.let {
                            Either.Right(it)
                        }
            false -> Either.Left(MatrixError.IncompatibleMatrixDimensions)
        }
    }

    private fun splitIntoChunks(
        columnCount: Int,
        rowCount: Int,
        inputItems: List<T>,
        originalColumnCount: Int
    ): List<List<T>> {
        return splitIntoRows(inputItems, originalColumnCount)
            .map { splitIntoRows(it, columnCount) }
            .chunked(rowCount)
            .map {
                it.foldRight((0..(originalColumnCount/columnCount) - 1).map { emptyList<T>() }, { columns, final ->
                    final.zip(columns, { column, value -> value + column })
                })
            }.flatMap {
                it
            }
    }

    private fun splitIntoRows(
        inputs: List<T>,
        columns: Int
    ) = inputs
        .chunked(columns)

    private fun apply(f:(T,T) -> T, other:Matrix<T>):Matrix<T> = items.zip(other.items,{x,y -> f(x,y)}).let { this.copy(items = it) }

    companion object {

        fun <T> builder():MatrixBuilder<T> = MatrixBuilder(this::from)

        private fun <T> from( addFunction:(T, T) -> T,
                      subtractFunction: (T,T) -> T,
                      multiplicationFunction: (T,T) -> T,
                      rows:Int,
                      columns:Int,
                      items:List<T>):Either<MatrixError, Matrix<T>>{
            return when{
                rows*columns == items.size ->
                    Matrix(addFunction, subtractFunction, multiplicationFunction, rows, columns, items).let {
                        Either.Right(it)
                    }
                else -> Either.Left(MatrixError.InputSizeNotMatchingWithMatrixSize)
            }

        }
    }

}