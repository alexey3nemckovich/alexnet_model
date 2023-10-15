package com.alexnet.data.repository

import com.alexnet.data.remote.ChatbotApi
import com.alexnet.domain.repository.ChatbotRepository
import com.alexnet.domain.repository.SendMessageResponse
import com.alexnet.util.ApiUtils.Companion.executeApiRequest
import javax.inject.Inject

class ChatbotRepositoryImpl @Inject constructor(
    private val chatbotApi: ChatbotApi
) : ChatbotRepository {

    override suspend fun sendMessage(message: String): SendMessageResponse {
        return executeApiRequest(
            request = null,
            apiCall = { _ ->
                chatbotApi.sendMessage(message)
            },
            parseResponse = {
                it.body()!!
            }
        )
    }
}
