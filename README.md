# AlexNet Chat bot app
Simple chat bot application

## Applicaton architecture
![architecture](images/architecture.jpg)


## Application components

### Android client
Uses all ML modules:
- ASR module to convert speech to text
- Seq2Seq module to generate text response
- TTS module to generate speech audio for generated response
![android_client](images/android_client.png)
### ASR module
HuBERT
- HUBERT_ASR_XLARGE
- pre-trained on 60,000 hours of unlabeled audio from Libri-Light dataset
- fine-tuned for ASR on 960 hours of transcribed audio from LibriSpeech dataset
![hubert_architecture](images/hubert_arch.png)
### Seq2Seq module
Encoder-Decoder with Attention-based mechanism
![hubert_architecture](images/encoder_decoder_with_attn_arch.png)
### TTS module
Tacotron trained on LJ Speech Dataset
![hubert_architecture](images/tacotron_arch.png)

