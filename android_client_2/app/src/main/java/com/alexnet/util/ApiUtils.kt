package com.alexnet.util

import android.util.Log
import com.alexnet.domain.model.Response
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONException
import org.json.JSONObject

class ApiUtils {

    companion object {

        suspend fun <T, R, E> executeApiRequest(
            request: T,
            apiCall: suspend (request: T) -> retrofit2.Response<R>,
            parseResponse: (response: retrofit2.Response<R>) -> E
        ): Response<E> {
            return withContext(Dispatchers.IO) {
                try {
                    val response = apiCall(request)

                    if (response.isSuccessful) {
                        Response.Success(parseResponse(response))
                    } else {
                        val errorBodyJson = response.errorBody()?.string()
                        val errorMessage = try {
                            val jsonObject = JSONObject(errorBodyJson)
                            jsonObject.getString("error")
                        } catch (e: JSONException) {
                            Log.e(Constants.APP_TAG, "Failed to parse json object: ${e.message}", e)

                            "Request failed with an unknown error"
                        }

                        Response.Failure(Exception(errorMessage))
                    }
                } catch (e: Exception) {
                    Log.e(Constants.APP_TAG, "API call failed: ${e.message}", e)

                    Response.Failure(e)
                }
            }
        }
    }
}
