package com.alexnet.data.remote

import com.alexnet.domain.model.TranscriptionOutput
import okhttp3.MultipartBody
import retrofit2.Response
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part

interface AsrmApi {

    @Multipart
    @POST("transcription")
    suspend fun getAudioTranscription(
        @Part file: MultipartBody.Part
    ): Response<TranscriptionOutput>

}
