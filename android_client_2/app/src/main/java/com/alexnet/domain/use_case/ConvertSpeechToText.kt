package com.alexnet.domain.use_case

import com.alexnet.domain.repository.AsrmRepository
import java.io.File

class ConvertSpeechToText(
    private val repo: AsrmRepository
) {
    suspend operator fun invoke(
        speechAudio: File
    ) = repo.convertSpeechToText(speechAudio)
}