package com.alexnet.presentation.screens.welcome.components

import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.size
import androidx.compose.material.CircularProgressIndicator
import androidx.compose.material.MaterialTheme
import androidx.compose.material.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import com.alexnet.domain.model.Response
import com.alexnet.presentation.MainViewModel


@Composable
fun AppLoadingProgressIndicator(
    viewModel: MainViewModel = hiltViewModel()
) {
    val text = "Welcome to AlexNet Chat bot!"

    Row(
        verticalAlignment = Alignment.CenterVertically
    ) {
        CircularProgressIndicator(
            modifier = Modifier.size(12.dp),
            color = Color.Gray,
            strokeWidth = 2.dp
        )

        Spacer(modifier = Modifier.size(4.dp))

        Text(
            text = text,
            style = MaterialTheme.typography.caption
        )
    }
}
