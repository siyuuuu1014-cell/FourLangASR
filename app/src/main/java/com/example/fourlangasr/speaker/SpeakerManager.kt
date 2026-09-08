package com.example.fourlangasr.speaker

import android.content.res.AssetManager

class SpeakerManager(
    assetManager: AssetManager,
) {

    private val engine =
        SpeakerEngine(
            assetManager
        )

    private val profiles =
        mutableListOf<
                SpeakerProfile
                >()

    @Synchronized
    fun load() {

        engine.load()
    }

    @Synchronized
    fun resetProfiles() {

        profiles.clear()
    }

    @Synchronized
    fun identify(
        samples: FloatArray,
        sampleRate: Int,
    ): SpeakerResult {

        if (
            samples.isEmpty()
        ) {

            return skippedResult(
                reason = "SKIP(empty)"
            )
        }

        val durationSec =
            samples.size /
                    sampleRate.toFloat()

        /*
         * PC benchmark mostly used multi-second speech.
         * Very short VAD segments are still sent to ASR,
         * but are not allowed to create/update a speaker profile.
         */
        if (
            durationSec <
            MIN_SPEAKER_AUDIO_SEC
        ) {

            return skippedResult(
                reason =
                    "SKIP(<" +
                            MIN_SPEAKER_AUDIO_SEC +
                            "s)"
            )
        }

        val embedding =
            try {

                engine.extract(
                    samples = samples,
                    sampleRate = sampleRate,
                )

            } catch (
                e: Throwable
            ) {

                /*
                 * Speaker failure must not kill ASR.
                 * MainActivity can still transcribe this utterance.
                 */
                return SpeakerResult(
                    label = "Speaker ?",
                    speakerId = null,
                    bestScore = Float.NaN,
                    secondScore = Float.NaN,
                    margin = Float.NaN,
                    isNewSpeaker = false,
                    acceptedExisting = false,
                    skipped = true,
                    profileCount =
                        profiles.size,
                    reason =
                        "ERROR(" +
                                e.javaClass
                                    .simpleName +
                                ")",
                )
            }

        /*
         * First valid utterance in a new conversation.
         */
        if (
            profiles.isEmpty()
        ) {

            return createNewProfile(
                embedding = embedding,
                bestScore = Float.NaN,
                secondScore = Float.NaN,
                margin = Float.NaN,
            )
        }

        val ranked =
            profiles
                .map { profile ->

                    profile to
                            profile
                                .similarity(
                                    embedding
                                )
                }
                .sortedByDescending {
                    it.second
                }

        val bestProfile =
            ranked[0].first

        val bestScore =
            ranked[0].second

        val secondScore =
            if (
                ranked.size >= 2
            ) {

                ranked[1].second

            } else {

                Float.NaN
            }

        val margin =
            if (
                secondScore.isFinite()
            ) {

                bestScore -
                        secondScore

            } else {

                Float.NaN
            }

        /*
         * With only one existing profile there is no Top-2,
         * so margin cannot be computed.
         *
         * In v0.3 observation mode we use score only until
         * a second profile exists. Once >=2 profiles exist,
         * both the PC-calibrated score and margin are required.
         */
        val acceptedExisting =
            if (
                profiles.size == 1
            ) {

                bestScore >=
                        SCORE_THRESHOLD

            } else {

                bestScore >=
                        SCORE_THRESHOLD &&
                        margin >=
                        MARGIN_THRESHOLD
            }

        if (
            acceptedExisting
        ) {

            /*
             * Only a high-confidence match may update a profile.
             */
            bestProfile.update(
                embedding
            )

            return SpeakerResult(
                label =
                    "Speaker " +
                            bestProfile.id,
                speakerId =
                    bestProfile.id,
                bestScore =
                    bestScore,
                secondScore =
                    secondScore,
                margin =
                    margin,
                isNewSpeaker =
                    false,
                acceptedExisting =
                    true,
                skipped =
                    false,
                profileCount =
                    profiles.size,
                reason =
                    if (
                        profiles.size == 1
                    ) {
                        "MATCH(score)"
                    } else {
                        "MATCH(score+margin)"
                    },
            )
        }

        /*
         * v0.3 observation-mode bootstrap:
         *
         * A rejected utterance creates the next profile while
         * fewer than MAX_SPEAKERS exist.
         *
         * This is intentionally visible/debuggable. If phone tests
         * show duplicate speakers caused by low-confidence utterances,
         * the next revision should add a tentative-speaker stage
         * instead of immediately promoting every rejection.
         */
        if (
            profiles.size <
            MAX_SPEAKERS
        ) {

            return createNewProfile(
                embedding = embedding,
                bestScore = bestScore,
                secondScore = secondScore,
                margin = margin,
            )
        }

        /*
         * Already have 4 profiles and this utterance does not
         * confidently match any of them.
         */
        return SpeakerResult(
            label = "Unknown",
            speakerId = null,
            bestScore = bestScore,
            secondScore = secondScore,
            margin = margin,
            isNewSpeaker = false,
            acceptedExisting = false,
            skipped = false,
            profileCount =
                profiles.size,
            reason =
                "UNKNOWN(max-profiles)",
        )
    }

    @Synchronized
    fun release() {

        profiles.clear()
        engine.release()
    }

    private fun createNewProfile(
        embedding: FloatArray,
        bestScore: Float,
        secondScore: Float,
        margin: Float,
    ): SpeakerResult {

        val id =
            profiles.size + 1

        val profile =
            SpeakerProfile(
                id = id,
                initialEmbedding =
                    embedding,
            )

        profiles.add(
            profile
        )

        return SpeakerResult(
            label =
                "Speaker $id",
            speakerId =
                id,
            bestScore =
                bestScore,
            secondScore =
                secondScore,
            margin =
                margin,
            isNewSpeaker =
                true,
            acceptedExisting =
                false,
            skipped =
                false,
            profileCount =
                profiles.size,
            reason =
                "NEW",
        )
    }

    private fun skippedResult(
        reason: String,
    ): SpeakerResult {

        return SpeakerResult(
            label = "Speaker ?",
            speakerId = null,
            bestScore = Float.NaN,
            secondScore = Float.NaN,
            margin = Float.NaN,
            isNewSpeaker = false,
            acceptedExisting = false,
            skipped = true,
            profileCount =
                profiles.size,
            reason = reason,
        )
    }

    companion object {

        /*
         * EXP3 + EXP4 Android candidate.
         */
        const val SCORE_THRESHOLD =
            0.45f

        const val MARGIN_THRESHOLD =
            0.12f

        const val MAX_SPEAKERS =
            4

        /*
         * Short speech remains available to ASR but is excluded
         * from speaker enrollment/profile updates.
         */
        const val MIN_SPEAKER_AUDIO_SEC =
            1.0f
    }
}
