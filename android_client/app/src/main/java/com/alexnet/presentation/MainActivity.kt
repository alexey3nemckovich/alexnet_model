package com.alexnet.presentation

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.provider.Settings
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.ViewModelProvider
import com.alexnet.presentation.screens.MainScreen
import com.alexnet.util.Constants
import com.alexnet.util.SharedPrefsManager
import dagger.hilt.android.AndroidEntryPoint

@AndroidEntryPoint
class MainActivity : AppCompatActivity() {

    private lateinit var viewModel: MainViewModel

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        handlePermissionResult(isGranted)
    }

    private val appSettingsLauncher =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) {
            checkPermissionsStatus()
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        viewModel = ViewModelProvider(this).get(MainViewModel::class.java)
        viewModel.setAudioRecordingOutputFilePath(externalCacheDir?.absolutePath + "/" + Constants.DEFAULT_AUDIO_FILE_NAME)

        setupViews()

        checkPermissionsStatus()
    }

    private fun setupViews() {
        setContent {
            MainScreen(
                requestMicrophonePermissions = { requestMicrophonePermission() },
                openAppPermissionsSettings = { navigateToAppSettings() },
            )
        }
    }

    private fun checkPermissionsStatus() {
        val status = getPermissionStatus()

        if (MainViewModel.PermissionsStatus.NOT_INITIALIZED == status) {
            requestPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
        } else {
            handlePermissionResult(MainViewModel.PermissionsStatus.GRANTED == status)
        }
    }

    private fun getPermissionStatus(): MainViewModel.PermissionsStatus {
        return if (ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.RECORD_AUDIO
            ) == PackageManager.PERMISSION_GRANTED
        ) {
            MainViewModel.PermissionsStatus.GRANTED
        } else if (SharedPrefsManager.arePermissionsDenied()) {
            MainViewModel.PermissionsStatus.DENIED_FOREVER
        } else if (shouldShowRequestPermissionRationale(Manifest.permission.RECORD_AUDIO)) {
            MainViewModel.PermissionsStatus.SHOULD_SHOW_RATIONALE
        } else {
            MainViewModel.PermissionsStatus.NOT_INITIALIZED
        }
    }

    private fun handlePermissionResult(isGranted: Boolean) {
        if (isGranted) {
            SharedPrefsManager.setPermissionsDenied(false)
            viewModel.updatePermissionsStatus(MainViewModel.PermissionsStatus.GRANTED)
        } else {
            val shouldShowRationale =
                shouldShowRequestPermissionRationale(Manifest.permission.RECORD_AUDIO)

            if (!shouldShowRationale) {
                SharedPrefsManager.setPermissionsDenied(true)
                viewModel.updatePermissionsStatus(MainViewModel.PermissionsStatus.DENIED_FOREVER)
            } else {
                viewModel.updatePermissionsStatus(MainViewModel.PermissionsStatus.SHOULD_SHOW_RATIONALE)
            }
        }
    }

    private fun requestMicrophonePermission() {
        requestPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
    }

    private fun navigateToAppSettings() {
        val intent = Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS)
        val uri = Uri.fromParts("package", packageName, null)
        intent.data = uri
        appSettingsLauncher.launch(intent)
    }

}
