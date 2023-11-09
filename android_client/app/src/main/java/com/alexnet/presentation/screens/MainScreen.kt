package com.alexnet.presentation.screens

import androidx.compose.runtime.Composable
import androidx.hilt.navigation.compose.hiltViewModel
import com.alexnet.presentation.MainViewModel
import com.alexnet.presentation.screens.lineitems.MessangerScreen
import com.alexnet.presentation.screens.welcome.WelcomeScreen

@Composable
fun MainScreen(
    viewModel: MainViewModel = hiltViewModel()
) {
    when (viewModel.loadingStatus) {
        true -> WelcomeScreen()
        false -> MessangerScreen()
    }
}
