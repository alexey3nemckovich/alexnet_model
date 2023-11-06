package com.alexnet.data.repository

import com.alexnet.data.remote.ChatbotApi
import com.alexnet.domain.model.ResponseInput
import com.alexnet.domain.repository.ChatbotRepository
import com.alexnet.domain.repository.SendMessageResponse
import com.alexnet.util.ApiUtils.Companion.executeApiRequest
import javax.inject.Inject

class ChatbotRepositoryImpl @Inject constructor(
    private val chatbotApi: ChatbotApi
) : ChatbotRepository {

    override suspend fun getResponse(message: String): SendMessageResponse {
        return executeApiRequest(
            request = null,
            apiCall = { _ ->
                chatbotApi.getResponse(ResponseInput(message))
            },
            parseResponse = {
                it.body()!!.output
            }
        )
    }
}
