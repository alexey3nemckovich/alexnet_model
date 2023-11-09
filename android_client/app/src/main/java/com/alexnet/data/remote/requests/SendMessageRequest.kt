package com.alexnet.data.remote.requests

import kotlinx.serialization.Serializable

@Serializable
data class SendMessageRequest(
    val subscriberId: String,
    val establishmentId: String,
    val date: String,
    val position: String
)
