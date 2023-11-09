package com.alexnet.data.remote

import com.alexnet.domain.model.ResponseInput
import com.alexnet.domain.model.ResponseOutput
import retrofit2.Response
import retrofit2.http.Body
import retrofit2.http.POST

interface ChatbotApi {

    @POST("response")
    suspend fun getResponse(
        @Body input: ResponseInput
    ): Response<ResponseOutput>

}
