package com.alexnet.di

import android.app.Application
import android.content.Context
import com.google.gson.Gson
import com.google.gson.GsonBuilder
import com.alexnet.data.remote.AsrmApi
import com.alexnet.data.remote.ChatbotApi
import com.alexnet.data.repository.AsrmRepositoryImpl
import com.alexnet.data.repository.ChatbotRepositoryImpl
import com.alexnet.domain.repository.AsrmRepository
import com.alexnet.domain.repository.ChatbotRepository
import com.alexnet.domain.use_case.*
import com.alexnet.util.Constants
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import okhttp3.OkHttpClient
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import javax.inject.Named
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object AppModule {
    @Singleton
    @Provides
    fun provideContext(application: Application): Context = application.applicationContext

    @Provides
    @Singleton
    @Named("httpClient")
    fun provideHttpClient(): OkHttpClient = OkHttpClient.Builder().build()

    @Provides
    @Singleton
    @Named("gson")
    fun provideGson(): Gson = GsonBuilder()
        .setLenient()
        .create()

    @Provides
    @Singleton
    @Named("chatbot")
    fun provideChatbotRetrofit(
        @Named("gson") gson: Gson,
        @Named("httpClient") httpClient: OkHttpClient
    ): Retrofit = Retrofit.Builder()
        .baseUrl(Constants.BASE_CHATBOT_API_URL)
        .addConverterFactory(GsonConverterFactory.create(gson))
        .client(httpClient)
        .build()

    @Provides
    @Singleton
    @Named("asrm")
    fun provideAsrmRetrofit(
        @Named("gson") gson: Gson,
        @Named("httpClient") httpClient: OkHttpClient
    ): Retrofit = Retrofit.Builder()
        .baseUrl(Constants.BASE_ASRM_API_URL)
        //.baseUrl(Constants.BASE_ASRM_API_URL)
        .addConverterFactory(GsonConverterFactory.create(gson))
        .client(httpClient)
        .build()

    @Provides
    @Singleton
    fun provideChatbotApi(@Named("chatbot") retrofit: Retrofit): ChatbotApi =
        retrofit.create(ChatbotApi::class.java)

    @Provides
    @Singleton
    fun provideAsrmApi(@Named("asrm") retrofit: Retrofit): AsrmApi =
        retrofit.create(AsrmApi::class.java)

    @Provides
    @Singleton
    fun provideChatbotRepository(
        chatbotApi: ChatbotApi
    ): ChatbotRepository = ChatbotRepositoryImpl(chatbotApi)

    @Provides
    @Singleton
    fun provideAsrmRepository(
        asrmApi: AsrmApi
    ): AsrmRepository = AsrmRepositoryImpl(asrmApi)

    @Provides
    @Singleton
    fun provideUseCases(
        chatbotRepo: ChatbotRepository,
        asrmRepo: AsrmRepository
    ) = UseCases(
        getBotResponse = GetBotResponse(chatbotRepo),
        convertSpeechToText = ConvertSpeechToText(asrmRepo)
    )
}
