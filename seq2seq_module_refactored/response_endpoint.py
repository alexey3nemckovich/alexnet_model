import contextlib
from fastapi import FastAPI
from schemas import ChatbotModel, ResponseInput, ResponseOutput


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    chatbot_model.load_model()
    yield


app = FastAPI(lifespan=lifespan)
chatbot_model = ChatbotModel()


@app.post("/response", response_model=ResponseOutput)
async def get_response(input: ResponseInput):
    response = chatbot_model.generate_response(input)
    return response
