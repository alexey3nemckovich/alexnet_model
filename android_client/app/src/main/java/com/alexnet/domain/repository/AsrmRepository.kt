package com.alexnet.domain.repository

import com.alexnet.domain.model.Response
import com.alexnet.domain.model.TranscriptionOutput
import java.io.File


typealias ConvertSpeechToTextResponse = Response<String>

interface AsrmRepository {

    suspend fun convertSpeechToText(
        speechAudio: File
    ): ConvertSpeechToTextResponse

}
