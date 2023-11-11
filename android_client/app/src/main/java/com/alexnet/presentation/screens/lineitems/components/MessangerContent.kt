package com.alexnet.presentation.screens.lineitems.components

import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.PressInteraction
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.wrapContentWidth
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.material.ButtonDefaults
import androidx.compose.material.Icon
import androidx.compose.material.OutlinedButton
import androidx.compose.material.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.drawWithContent
import androidx.compose.ui.draw.scale
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.DrawStyle
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.res.vectorResource
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.hilt.navigation.compose.hiltViewModel
import com.alexnet.R
import com.alexnet.domain.model.Message
import com.alexnet.presentation.MainViewModel
import kotlinx.coroutines.delay

@Composable
fun MessangerContent(
    padding: PaddingValues,
    messages: List<Message>,
    sendMessage: (message: String) -> Unit,
    viewModel: MainViewModel = hiltViewModel(),
) {
    var message by remember { mutableStateOf(TextFieldValue("")) }

    Box(
        Modifier.fillMaxSize()
    ){


        Column(
            modifier = Modifier
                .fillMaxSize()
        ) {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f)
            ) {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(start = 8.dp, end = 8.dp)
                ){
                    LazyColumn(
                        modifier = Modifier
                            .fillMaxSize()
                    ) {
                        items(messages) { message ->
                            MessageItem(message)
                        }
                    }
                }

                if (viewModel.recordingAudio) {
                    val pulseAnim = rememberInfiniteTransition()
                    val scale by pulseAnim.animateFloat(
                        initialValue = 0.5f,
                        targetValue = 1.5f,
                        animationSpec = infiniteRepeatable(
                            animation = tween(durationMillis = 1000),
                            repeatMode = RepeatMode.Reverse
                        ), label = ""
                    )

                    Box(
                        modifier = Modifier
                            .fillMaxSize()
                            .background(Color(0x80000000)),
                        contentAlignment = Alignment.Center
                    ) {
                        Column(
                            horizontalAlignment = Alignment.CenterHorizontally
                        ) {
                            Box(
                                modifier = Modifier
                                    .size(100.dp)
                                    .scale(scale)
                                    .graphicsLayer {
                                        clip = true
                                    }
                                    .drawWithContent {
                                        val circleRadius = size.minDimension / 4
                                        drawCircle(
                                            color = Color.Red,
                                            radius = circleRadius,
                                            center = Offset(size.width / 2, size.height / 2)
                                        )
                                        val circleRadius2 = circleRadius + 10.0f
                                        drawCircle(
                                            color = Color.Black,
                                            style = Stroke(width=2.0f),
                                            radius = circleRadius2,
                                            center = Offset(size.width / 2, size.height / 2)
                                        )
                                    }
                            )
                            Text(
                                text = "Listening to you...",
                                color = Color.White,
                                fontSize = 16.sp
                            )
                        }
                    }
                }
            }

            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier
                    .fillMaxWidth()
                    .background(Color(0xFFE4E4E4))
                    .padding(vertical = 6.dp, horizontal = 8.dp)
                    .height(36.dp),
            ) {
                Box (
                    modifier = Modifier
                        .padding(0.dp)
                        .fillMaxSize()
                        .background(Color.Transparent)
                        .weight(1f),
                    contentAlignment = Alignment.CenterStart
                ){
                    BasicTextField(
                        value = message,
                        onValueChange = {
                            message = it
                        },
                        singleLine = true,
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(0.dp),
                        textStyle = TextStyle(fontSize = 18.sp),
                    )

                    if (message.text.isEmpty()) {
                        Text(
                            text = "Message",
                            fontSize = 18.sp,
                            color = Color.Gray,
                            modifier = Modifier.wrapContentWidth()
                        )
                    }
                }

                val interactionSource = remember { MutableInteractionSource() }
                LaunchedEffect(interactionSource) {
                    interactionSource.interactions.collect { interaction ->
                        when (interaction) {
                            is PressInteraction.Press -> {
                                viewModel.startRecordingAudio()
                            }
                            is PressInteraction.Release -> {
                                viewModel.stopRecordingAudio()
                            }
                            is PressInteraction.Cancel -> {
                                viewModel.cancelRecordingAudio()
                            }
                        }
                    }
                }

                OutlinedButton(
                    colors = ButtonDefaults.outlinedButtonColors(backgroundColor = Color.Transparent),
                    border = BorderStroke(0.dp, Color.Transparent),
                    interactionSource = interactionSource,
                    onClick = {}
                ) {
                    Icon(
                        imageVector = ImageVector.vectorResource(id= R.drawable.baseline_mic_24),
                        contentDescription = null,
                        tint = Color.White
                    )
                }
                OutlinedButton(
                    colors = ButtonDefaults.outlinedButtonColors(backgroundColor = Color.Transparent),
                    border = BorderStroke(0.dp, Color.Transparent),
                    onClick = {
                        val currentMessage = message.text
                        sendMessage(currentMessage)

                        message = TextFieldValue("")
                    },
                ) {
                    Icon(
                        imageVector = ImageVector.vectorResource(id= R.drawable.ic_send_message),
                        contentDescription = null,
                        tint = Color.White
                    )
                }
            }
        }
    }
}
