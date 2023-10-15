package com.alexnet.domain.repository

import com.alexnet.domain.model.Response

typealias SendMessageResponse = Response<String>

interface ChatbotRepository {

    suspend fun sendMessage(
        message: String
    ): SendMessageResponse

}
