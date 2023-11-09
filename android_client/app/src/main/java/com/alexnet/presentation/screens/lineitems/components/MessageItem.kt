package com.alexnet.presentation.screens.lineitems.components

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.alexnet.domain.model.Message

@Composable
fun MessageItem(
    message: Message
) {
    Box(
        modifier = Modifier
            .fillMaxSize(),
        contentAlignment = if (message.bot) Alignment.TopStart else Alignment.TopEnd
    ) {
        Box(
            modifier = Modifier
                .background(
                    color = Color(if (message.bot) 0xFF26A69A else 0xFFFF7043),
                    shape = RoundedCornerShape(
                        topStart = if (message.bot) 0.dp else 20.dp,
                        topEnd = if (message.bot) 20.dp else 0.dp,
                        bottomEnd = 20.dp,
                        bottomStart = 20.dp
                    )
                ),
        ) {
            Text(
                text = message.value,
                color = Color.White,
                fontSize = 14.sp,
                modifier = Modifier.padding(14.dp)
            )
        }
    }
}
