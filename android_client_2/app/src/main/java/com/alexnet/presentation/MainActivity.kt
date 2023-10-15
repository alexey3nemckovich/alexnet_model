package com.alexnet.presentation

import android.os.Bundle
import androidx.activity.compose.setContent
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.ViewModelProvider
import com.alexnet.presentation.screens.MainScreen
import com.alexnet.util.Constants
import dagger.hilt.android.AndroidEntryPoint

@AndroidEntryPoint
class MainActivity : AppCompatActivity() {

    private lateinit var viewModel: MainViewModel

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        viewModel = ViewModelProvider(this).get(MainViewModel::class.java)
        viewModel.setFilePath(externalCacheDir?.absolutePath + "/" + Constants.DEFAULT_AUDIO_FILE_NAME)

        setupViews()
    }

    private fun setupViews() {
        setContent {
            MainScreen()
        }
    }

}
