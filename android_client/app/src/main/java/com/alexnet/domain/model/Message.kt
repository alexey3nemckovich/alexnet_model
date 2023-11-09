package com.alexnet.domain.model

import kotlinx.serialization.Serializable

@Serializable
data class Message(
    val bot: Boolean,
    val value: String
)
