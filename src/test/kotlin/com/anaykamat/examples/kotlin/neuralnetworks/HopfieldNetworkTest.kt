package com.anaykamat.examples.kotlin.neuralnetworks

import org.junit.Assert
import org.junit.Test

class HopfieldNetworkTest {

    @Test
    fun shouldBuildHopfieldNetworkWithDefaultWeightsForGivenInputSize(){
        val hopfieldNetwork = HopfieldNetwork.forInputSize(3)
        Assert.assertEquals(
            listOf(0,0,0,0,0,0,0,0,0),
            hopfieldNetwork.weights
        )
    }

    @Test
    fun trainDefinesWeightOfEachCellAsProductOfValuesOfNodesMarkedByItsRowsAndColumnsWhileKeepingDiagonalZero(){
        val hopfieldNetwork = HopfieldNetwork.forInputSize(4)
        val trainedHopfieldNetwork: HopfieldNetwork = HopfieldNetwork.train(hopfieldNetwork, listOf(1, -1, 1, -1))

        Assert.assertEquals(
            listOf(0, -1, 1, -1, -1, 0, -1,  1, 1,  -1, 0,  -1, -1, 1, -1,  0),
            trainedHopfieldNetwork.weights
        )
    }

    @Test
    fun trainShouldAddNewWeightsToExistingWeights(){
        val hopfieldNetwork = HopfieldNetwork.forInputSize(4).copy(weights = listOf(0, -1, 1, -1, -1, 0, -1,  1, 1,  -1, 0,  -1, -1, 1, -1,  0))
        val trainedHopfieldNetwork: HopfieldNetwork = HopfieldNetwork.train(hopfieldNetwork, listOf(1, -1, 1, -1))

        Assert.assertEquals(
            listOf(0, -2, 2, -2, -2, 0, -2,  2, 2,  -2, 0,  -2, -2, 2, -2,  0),
            trainedHopfieldNetwork.weights
        )
    }

    @Test
    fun thinkShouldIdentifyTheProperPatternFromTheTrainedPatterns(){
        val hopfieldNetwork = HopfieldNetwork.forInputSize(4).let {
            HopfieldNetwork.train(it, listOf(1,1,1,-1))
        }

        val output:List<Int> = hopfieldNetwork.think(listOf(-1,1,1,-1))

        Assert.assertEquals(
            listOf(1,1,1,-1),
            output
        )
    }


    @Test
    fun testWithRealPatterns(){
        val pattern =
          listOf(listOf("O O O O O ",
                        " O O O O O",
                        "O O O O O ",
                        " O O O O O",
                        "O O O O O ",
                        " O O O O O",
                        "O O O O O ",
                        " O O O O O",
                        "O O O O O ",
                        " O O O O O"  ),

            listOf( "OO  OO  OO",
                    "OO  OO  OO",
                    "  OO  OO  ",
                    "  OO  OO  ",
                    "OO  OO  OO",
                    "OO  OO  OO",
                    "  OO  OO  ",
                    "  OO  OO  ",
                    "OO  OO  OO",
                    "OO  OO  OO"  ),

            listOf( "OOOOO     ",
                    "OOOOO     ",
                    "OOOOO     ",
                    "OOOOO     ",
                    "OOOOO     ",
                    "     OOOOO",
                    "     OOOOO",
                    "     OOOOO",
                    "     OOOOO",
                    "     OOOOO"  ),

            listOf( "O  O  O  O",
                    " O  O  O  ",
                    "  O  O  O ",
                    "O  O  O  O",
                    " O  O  O  ",
                    "  O  O  O ",
                    "O  O  O  O",
                    " O  O  O  ",
                    "  O  O  O ",
                    "O  O  O  O"  ),

            listOf( "OOOOOOOOOO",
                    "O        O",
                    "O OOOOOO O",
                    "O O    O O",
                    "O O OO O O",
                    "O O OO O O",
                    "O O    O O",
                    "O OOOOOO O",
                    "O        O",
                    "OOOOOOOOOO"  ) ).map { pattern -> pattern.map { it.toCharArray().map { if (it == 'O') 1 else -1 } }.flatten()  }

        val hopfieldNetwork = HopfieldNetwork.forInputSize(100)
        val trainedNetwork = pattern.fold(hopfieldNetwork, {network, pattern -> HopfieldNetwork.train(network, pattern)})

        val patternToTest = listOf( "OOOOOOOOOO",
                                    "O        O",
                                    "O        O",
                                    "O        O",
                                    "O   OO   O",
                                    "O   OO   O",
                                    "O        O",
                                    "O        O",
                                    "O        O",
                                    "OOOOOOOOOO"  ).map { it.toCharArray().map { if (it == 'O') 1 else -1 } }.flatten()

        val identifiedPattern = trainedNetwork.think(patternToTest)

        Assert.assertEquals(pattern.last(), identifiedPattern)
    }




}