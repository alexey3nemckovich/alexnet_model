import contextlib

from fastapi import FastAPI, UploadFile, File, HTTPException

from schemas import AsrModel, TranscriptionOutput


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    asr_model.load_model()
    yield


app = FastAPI(lifespan=lifespan)
asr_model = AsrModel()


@app.post("/transcription", response_model=TranscriptionOutput)
async def get_transcription(file: UploadFile = File(...)):
    try:
        response = asr_model.get_audio_transcription(file)
        print(response.text)
        return response
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
