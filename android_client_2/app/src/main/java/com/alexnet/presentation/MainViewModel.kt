package com.alexnet.presentation

import android.media.MediaPlayer
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.alexnet.AlexNetApp
import com.alexnet.domain.model.Message
import com.alexnet.domain.model.Response.Loading
import com.alexnet.domain.model.Response.Success
import com.alexnet.domain.repository.ConvertSpeechToTextResponse
import com.alexnet.domain.repository.ConvertTextToSpeechResponse
import com.alexnet.domain.repository.SendMessageResponse
import com.alexnet.domain.use_case.UseCases
import com.alexnet.util.AudioRecorder
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.MultipartBody
import okhttp3.ResponseBody
import okhttp3.ResponseBody.Companion.toResponseBody
import java.io.File
import javax.inject.Inject

@HiltViewModel
class MainViewModel @Inject constructor(
    private val useCases: UseCases
) : ViewModel() {

    var messages by mutableStateOf<List<Message>>(ArrayList())
        private set
    var spellResponse by mutableStateOf(true)
        private set
    var sendMessageResponse by mutableStateOf<SendMessageResponse>(Success(""))
        private set
    var convertSpeechToTextResponse by mutableStateOf<ConvertSpeechToTextResponse>(Success(""))
        private set
    var convertTextToSpeechResponse by mutableStateOf<ConvertTextToSpeechResponse>(Success("".toResponseBody(null)))
        private set

    var loadingStatus by mutableStateOf(true)
        private set

    var recordingAudio by mutableStateOf<Boolean>(false)
        private set

    private val audioRecorder = AudioRecorder()

    enum class PermissionsStatus {
        NOT_INITIALIZED,
        SHOULD_SHOW_RATIONALE,
        DENIED_FOREVER,
        GRANTED
    }

    //var appStart = true

    init {
        viewModelScope.launch {
            delay(3000)

            loadingStatus = false
        }
    }

    var outputFilePath: String = ""

    fun setFilePath(filePath: String){
        outputFilePath = filePath
    }

    fun startRecordingAudio() = viewModelScope.launch{
        recordingAudio = true

        audioRecorder.startRecording(outputFilePath)
    }

    fun stopRecordingAudio() = viewModelScope.launch{
        audioRecorder.stopRecording()
        recordingAudio = false

        val file = File(outputFilePath)
        convertSpeechToText(file)
    }

    fun cancelRecordingAudio() = viewModelScope.launch{
        audioRecorder.cancelRecording()
        recordingAudio = false
    }

    fun checkUpdates() = viewModelScope.launch(Dispatchers.Default) {
        //AppCheckUtils.updateAppCheckToken()
        //DeviceUtils.updateDeviceId()
        //updateLineItems()
    }

    private fun updateLoadingStatus() {
//        loadingStatus =
//            permissionsStatus != PermissionsStatus.GRANTED || appInstanceRegistered != true || _loadingEstablishments.value || initializingCurrentEstablishment
    }

    private fun initApp() {
        //updateAppInstanceRegistrationStatus(SharedPrefsManager.isAppRegistered())
        //updateFCMToken()
        //getEstablishments()
        //getLineItems()
    }

    fun updateSpellSwitchState(turn: Boolean) = viewModelScope.launch{
        spellResponse = turn
    }

    fun sendMessage(message: String) = viewModelScope.launch(Dispatchers.Default) {
        sendMessageResponse = Loading
        sendMessageResponse = useCases.getBotResponse(message)

        when (val response = sendMessageResponse){
            // delete file
            is Success -> {
                val list = ArrayList(messages)
                list.add(Message(false, message))
                list.add(Message(true, response.data))

                messages = list

                if (spellResponse){
                    convertTextToSpeechResponse = Loading
                    convertTextToSpeechResponse = useCases.convertTextToSpeech(response.data)

                    when (val speechResponse = convertTextToSpeechResponse){
                        is Success -> {
                            val mediaPlayer = MediaPlayer()
                            val audioData = speechResponse.data.bytes()

                            val tempAudioFile =
                                withContext(Dispatchers.IO) {
                                    File.createTempFile(
                                        "tempAudio",
                                        ".mp3",
                                        AlexNetApp.appContext.cacheDir
                                    )
                                }

                            tempAudioFile.writeBytes(audioData)

                            mediaPlayer.setDataSource(tempAudioFile.absolutePath)
                            mediaPlayer.prepare()
                            mediaPlayer.start()
                        }
                        else -> {}
                    }
                }
            }
            else -> {}
        }
    }

    private fun convertSpeechToText(speechAudio: File) = viewModelScope.launch(Dispatchers.Default) {
//        val mediaPlayer = MediaPlayer()
//        mediaPlayer.setDataSource(speechAudio.path)
//        mediaPlayer.prepare()
//        mediaPlayer.start()
        convertSpeechToTextResponse = Loading
        convertSpeechToTextResponse = useCases.convertSpeechToText(speechAudio)

        when(val response = convertSpeechToTextResponse){
            is Success -> {
                sendMessage(response.data)
            }
            else -> {}
        }
    }

}
