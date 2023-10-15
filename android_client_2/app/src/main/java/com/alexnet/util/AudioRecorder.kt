package com.alexnet.util

import com.github.squti.androidwaverecorder.WaveRecorder
import java.io.File

class AudioRecorder {
    private var waveRecorder: WaveRecorder? = null
    private var filePath: String? = null

    fun startRecording(outputFilePath: String) {
        filePath = outputFilePath
        waveRecorder = WaveRecorder(outputFilePath)
        waveRecorder!!.startRecording()
    }

    fun stopRecording() {
        waveRecorder?.stopRecording()
    }

    fun cancelRecording(){
        stopRecording()

        filePath?.let {
            val fileToDelete = File(it)
            if (fileToDelete.exists()) {
                fileToDelete.delete()
            }
        }
    }
}
