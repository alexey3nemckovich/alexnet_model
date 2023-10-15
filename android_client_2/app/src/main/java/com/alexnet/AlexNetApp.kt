package com.alexnet

import android.app.Application
import android.content.Context
import com.alexnet.util.SharedPrefsManager
import dagger.hilt.android.HiltAndroidApp

@HiltAndroidApp
class AlexNetApp : Application() {
    companion object {
        lateinit var appContext: Context
    }

    override fun onCreate() {
        super.onCreate()

        appContext = applicationContext
        SharedPrefsManager.init(appContext)
    }

}
