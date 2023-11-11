import contextlib

from fastapi import FastAPI, HTTPException
from schemas import TtsModel, SpeechInput
from fastapi.responses import FileResponse


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    tts_model.load_model()
    yield


app = FastAPI(lifespan=lifespan)
tts_model = TtsModel()


@app.post("/speech")
async def generate_speech(input: SpeechInput):
    try:
        speech_file_path = tts_model.generate_speech(input)
        print(speech_file_path)
        return FileResponse(speech_file_path)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
