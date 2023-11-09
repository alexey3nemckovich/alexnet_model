package com.alexnet.presentation.screens.lineitems

import androidx.compose.material.Scaffold
import androidx.compose.runtime.Composable
import androidx.hilt.navigation.compose.hiltViewModel
import com.alexnet.components.TopBar
import com.alexnet.presentation.MainViewModel
import com.alexnet.presentation.screens.lineitems.components.MessangerContent
import com.alexnet.presentation.screens.lineitems.components.GenerateResponseProgress
import com.alexnet.presentation.screens.lineitems.components.ConvertSpeechToTextProgress

@Composable
fun MessangerScreen(
    viewModel: MainViewModel = hiltViewModel()
) {
    Scaffold(
        topBar = {
            TopBar(
                switchState = viewModel.spellResponse,
                onSwitchStateChanged = { isChecked ->
                    viewModel.updateSpellSwitchState(isChecked)
                }
            )
        },
        content = { padding ->
            MessangerContent(
                padding = padding,
                messages = viewModel.messages,
                sendMessage = { message ->
                    viewModel.sendMessage(message)
                }
            )
        }
    )

    GenerateResponseProgress()
    ConvertSpeechToTextProgress()
}
