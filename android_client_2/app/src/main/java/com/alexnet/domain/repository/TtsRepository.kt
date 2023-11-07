package com.alexnet.domain.repository

import com.alexnet.domain.model.Response
import okhttp3.MultipartBody
import okhttp3.ResponseBody


typealias ConvertTextToSpeechResponse = Response<ResponseBody>

interface TtsRepository {

    suspend fun convertTextToSpeech(
        text: String
    ): ConvertTextToSpeechResponse

}
