package com.alexnet.data.remote.responses

import kotlinx.serialization.Serializable

@Serializable
data class BaseResponse(
    val message: String
)