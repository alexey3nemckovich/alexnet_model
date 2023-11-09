package com.alexnet.data.remote.requests

import kotlinx.serialization.Serializable

@Serializable
data class ConvertSpeechToTextRequest(
    val subscriberId: String,
    val lineItemId: String
)
