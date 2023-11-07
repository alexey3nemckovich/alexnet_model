package com.alexnet.data.repository

import com.alexnet.data.remote.TtsApi
import com.alexnet.domain.model.SpeechInput
import com.alexnet.domain.repository.ConvertTextToSpeechResponse
import com.alexnet.domain.repository.TtsRepository
import com.alexnet.util.ApiUtils.Companion.executeApiRequest
import javax.inject.Inject

class TtsRepositoryImpl @Inject constructor(
    private val ttsApi: TtsApi
) : TtsRepository {

    override suspend fun convertTextToSpeech(text: String): ConvertTextToSpeechResponse {
        return executeApiRequest(
            request = null,
            apiCall = { _ ->
                ttsApi.getSpeechAudio(SpeechInput(text))
            },
            parseResponse = {
                it.body()!!
            }
        )
    }
}
