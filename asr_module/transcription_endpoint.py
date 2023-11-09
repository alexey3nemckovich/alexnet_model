import contextlib
import shutil

from fastapi import FastAPI, UploadFile, File
from schemas import AsrModel, TranscriptionOutput


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    asr_model.load_model()
    yield


app = FastAPI(lifespan=lifespan)
asr_model = AsrModel()


#, response_model=TranscriptionOutput
@app.post("/transcription")
async def get_transcription(file: UploadFile = File(...)):
    # print("request get file")
    #
    # # Define the local path where you want to save the file
    # local_path = f"/home/alex/1.wav"
    #
    # # Copy the contents of the uploaded file to the local path
    # with open(local_path, "wb") as local_file:
    #     shutil.copyfileobj(file.file, local_file)
    #
    # # Optionally, close and delete the temporary file
    # file.file.close()
    #
    # return "good"  ## response.output

    print("1")
    response = asr_model.get_audio_transcription(file)
    print("2")
    print(response.output)
    print("3")
    return response.output
    #
    # print("request get file")
    #
    # # Define the local path where you want to save the file
    # local_path = f"/home/alex/1.wav"
    #
    # # Copy the contents of the uploaded file to the local path
    # with open(local_path, "wb") as local_file:
    #     shutil.copyfileobj(file.file, local_file)
    #
    # # Optionally, close and delete the temporary file
    # file.file.close()
    #
    # return "good"
