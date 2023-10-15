package com.alexnet.data.remote

import okhttp3.MultipartBody
import okhttp3.ResponseBody
import retrofit2.Call
import retrofit2.Response
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part

interface AsrmApi {

    @Multipart
    @POST("convertSpeechToText")
    suspend fun convertSpeechToText(
        @Part file: MultipartBody.Part
    ): Response<String>

}
