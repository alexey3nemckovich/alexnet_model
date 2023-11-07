package com.alexnet.util

object Constants {
    // API URLs
    const val BASE_CHATBOT_API_URL =
        "http://10.0.2.2:8000/"
    const val BASE_ASRM_API_URL =
        "http://10.0.2.2:8080/"
    const val BASE_TTS_API_URL =
        "http://10.0.2.2:8088/"


    //App
    const val APP_TAG = "AlexNet"

    //Realtime database
    const val SUBSCRIBERS = "subscribers"
    const val LINE_ITEMS = "lineItems"
    const val ESTABLISHMENT = "establishment"
    const val ESTABLISHMENTS = "establishments"
    const val ESTABLISHMENTS_LINE_ITEMS = "establishmentsLineItems"

    //Actions
    const val ENTER_POSITION_TO_TRACK = "Enter position to track"
    const val SELECT_ESTABLISHMENT = "Select establishment"
    const val UNSUBSCRIBE_FROM_LINE_ITEM = "Unsubscribe from line item"

    //Buttons
    const val SUBSCRIBE = "Subscribe"
    const val DISMISS = "Cancel"
    const val CHANGE = "Change"
    const val CONFIRM = "Confirm"
    const val SELECT_LOCATION_ON_MAP = "Select location on map"

    //Placeholders
    const val BOOK_TITLE = "Type a book title..."
    const val AUTHOR = "Type the author name..."
    const val NO_VALUE = ""

    //Shared prefs
    const val SHARED_PREFS_NAME = "com.alexnet.data"
    const val APP_REGISTERED_PREF = "APP_REGISTERED"
    const val PERMISSIONS_DENIED_PREF = "PERMISSIONS_DENIED"
    const val FCM_TOKEN_PREF = "FCM_TOKEN"
    const val FCM_TOKEN_OUTDATED_PREF = "FCM_TOKEN_OUTDATED"
    const val DEVICE_ID_PREF = "DEVICE_ID"
    const val APP_CHECK_TOKEN_PREF = "APP_CHECK_TOKEN"
    const val APP_CHECK_TOKEN_EXPIRY_TIME_PREF = "APP_CHECK_TOKEN_EXPIRY_TIME"

    //Intent actions
    const val NEW_FCM_TOKEN = "NEW_FCM_TOKEN"
    const val LINE_ITEMS_LIST_UPDATED = "LINE_ITEMS_LIST_UPDATED"

    const val TOKEN = "TOKEN"

    const val DEFAULT_AUDIO_FILE_NAME = "speech.wav"
}
