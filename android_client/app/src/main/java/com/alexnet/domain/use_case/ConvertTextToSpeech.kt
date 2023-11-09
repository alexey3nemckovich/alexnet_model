package com.alexnet.domain.use_case

import com.alexnet.domain.repository.TtsRepository

class ConvertTextToSpeech(
    private val repo: TtsRepository
) {
    suspend operator fun invoke(
        text: String
    ) = repo.convertTextToSpeech(text)
}