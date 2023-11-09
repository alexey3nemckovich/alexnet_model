package com.alexnet.components

import androidx.compose.material.Switch
import androidx.compose.material.Text
import androidx.compose.material.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.MutableState
import androidx.compose.ui.res.stringResource
import com.alexnet.R

@Composable
fun TopBar(
    switchState: Boolean,
    onSwitchStateChanged: (Boolean) -> Unit
) {
    TopAppBar(
        title = {
            Text(
                text = stringResource(R.string.app_name)
            )
        },
        actions = {
            // Add a switch to the top bar
            Switch(
                checked = switchState,
                onCheckedChange = { isChecked ->
                    onSwitchStateChanged(isChecked)
                }
            )
        }
    )
}
