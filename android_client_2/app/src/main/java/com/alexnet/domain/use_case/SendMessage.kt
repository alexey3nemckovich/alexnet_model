package com.alexnet.domain.use_case

import com.alexnet.domain.repository.ChatbotRepository

class SendMessage(
    private val repo: ChatbotRepository
) {
    suspend operator fun invoke(
        message: String
    ) = repo.sendMessage(message)
}