package com.alexnet.data.remote

import com.alexnet.domain.model.ResponseInput
import com.alexnet.domain.model.ResponseOutput
import com.alexnet.domain.model.SpeechInput
import okhttp3.MultipartBody
import okhttp3.ResponseBody
import retrofit2.Response
import retrofit2.http.Body
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part

interface TtsApi {

    @POST("speech")
    suspend fun getSpeechAudio(
        @Body input: SpeechInput
    ): Response<ResponseBody>

}
