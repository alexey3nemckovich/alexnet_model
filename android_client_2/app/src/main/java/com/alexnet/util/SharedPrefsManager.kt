package com.alexnet.util

import android.content.Context
import android.content.SharedPreferences
import androidx.core.content.edit


class SharedPrefsManager(private val context: Context) {

    companion object {

        private lateinit var sharedPreferences: SharedPreferences

        fun init(context: Context) {
            sharedPreferences =
                context.getSharedPreferences(Constants.SHARED_PREFS_NAME, Context.MODE_PRIVATE)
        }

    }
}
