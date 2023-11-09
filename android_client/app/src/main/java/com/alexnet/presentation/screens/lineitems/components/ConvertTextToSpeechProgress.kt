package com.alexnet.presentation.screens.lineitems.components

import android.widget.Toast
import androidx.compose.runtime.Composable
import androidx.compose.ui.platform.LocalContext
import androidx.hilt.navigation.compose.hiltViewModel
import com.alexnet.components.ProgressBarDialog
import com.alexnet.domain.model.Response.Failure
import com.alexnet.domain.model.Response.Loading
import com.alexnet.presentation.MainViewModel

@Composable
fun ConvertTextToSpeechProgress(
    viewModel: MainViewModel = hiltViewModel()
) {
    when (val convertTextToSpeechResponse = viewModel.convertTextToSpeechResponse) {
        is Loading -> ProgressBarDialog("Converting text to speech...")
        is Failure -> {
            Toast.makeText(
                LocalContext.current,
                "Failed to convert speech to text: " + convertTextToSpeechResponse.e!!.message,
                Toast.LENGTH_LONG
            ).show()
            print(convertTextToSpeechResponse.e)
        }
        else -> {

        }
    }
}
