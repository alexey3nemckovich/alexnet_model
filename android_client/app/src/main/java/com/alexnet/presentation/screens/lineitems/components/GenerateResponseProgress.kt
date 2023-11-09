package com.alexnet.presentation.screens.lineitems.components

import android.widget.Toast
import androidx.compose.runtime.Composable
import androidx.compose.ui.platform.LocalContext
import androidx.hilt.navigation.compose.hiltViewModel
import com.alexnet.components.ProgressBarDialog
import com.alexnet.domain.model.Response.Failure
import com.alexnet.domain.model.Response.Loading
import com.alexnet.domain.model.Response.Success
import com.alexnet.presentation.MainViewModel

@Composable
fun GenerateResponseProgress(
    viewModel: MainViewModel = hiltViewModel()
) {
    when (val subscribeToLineItemResponse = viewModel.sendMessageResponse) {
        is Loading -> ProgressBarDialog("Getting response...")
        is Failure -> {
            Toast.makeText(
                LocalContext.current,
                "Failed to get response: " + subscribeToLineItemResponse.e!!.message,
                Toast.LENGTH_LONG
            ).show()
            print(subscribeToLineItemResponse.e)
        }
        else -> {

        }
    }
}
