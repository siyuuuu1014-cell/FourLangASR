package com.example.fourlangasr.speaker

internal class SpeakerProfile(
    val id: Int,
    initialEmbedding: FloatArray,
) {

    var centroid: FloatArray =
        normalize(initialEmbedding)
        private set

    var sampleCount: Int = 1
        private set

    fun similarity(
        embedding: FloatArray,
    ): Float {

        val normalized =
            normalize(embedding)

        var sum = 0.0f

        for (i in centroid.indices) {

            sum +=
                centroid[i] *
                        normalized[i]
        }

        return sum
    }

    fun update(
        embedding: FloatArray,
    ) {

        val normalized =
            normalize(embedding)

        require(
            normalized.size ==
                    centroid.size
        ) {
            "Embedding dimension mismatch: " +
                    "${normalized.size} vs " +
                    "${centroid.size}"
        }

        val oldCount =
            sampleCount.toFloat()

        val newCentroid =
            FloatArray(
                centroid.size
            )

        for (i in centroid.indices) {

            newCentroid[i] =
                (
                    centroid[i] *
                            oldCount +
                            normalized[i]
                    ) /
                        (oldCount + 1.0f)
        }

        centroid =
            normalize(
                newCentroid
            )

        sampleCount += 1
    }

    companion object {

        fun normalize(
            input: FloatArray,
        ): FloatArray {

            var sumSquares =
                0.0

            for (value in input) {

                sumSquares +=
                    value.toDouble() *
                            value.toDouble()
            }

            val norm =
                kotlin.math.sqrt(
                    sumSquares
                ).toFloat()

            require(
                norm > 1.0e-12f
            ) {
                "Speaker embedding norm is zero."
            }

            return FloatArray(
                input.size
            ) { i ->

                input[i] / norm
            }
        }
    }
}
