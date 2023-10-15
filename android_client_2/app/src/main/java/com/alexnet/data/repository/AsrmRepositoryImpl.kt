package com.alexnet.data.repository

import com.alexnet.data.remote.AsrmApi
import com.alexnet.domain.repository.AsrmRepository
import com.alexnet.domain.repository.ConvertSpeechToTextResponse
import com.alexnet.util.ApiUtils.Companion.executeApiRequest
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody
import okhttp3.ResponseBody
import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response
import java.io.File
import javax.inject.Inject

class AsrmRepositoryImpl @Inject constructor(
    private val asrmApi: AsrmApi
) : AsrmRepository {

    override suspend fun convertSpeechToText(speechAudio: File): ConvertSpeechToTextResponse {
        return executeApiRequest(
            request = null,
            apiCall = { _ ->
                val requestFile = RequestBody.create("multipart/form-data".toMediaTypeOrNull(), speechAudio)
                val body = MultipartBody.Part.createFormData("file", speechAudio.name, requestFile)
                asrmApi.convertSpeechToText(body)
            },
            parseResponse = {
                it.body()!!
            }
        )
    }
}
