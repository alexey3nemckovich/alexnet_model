import contextlib
from fastapi import FastAPI, HTTPException
from schemas import ChatbotModel, ResponseInput, ResponseOutput


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    chatbot_model.load_model()
    yield


app = FastAPI(lifespan=lifespan)
chatbot_model = ChatbotModel()


@app.post("/response", response_model=ResponseOutput)
async def get_response(input: ResponseInput):
    try:
        response = chatbot_model.generate_response(input)
        print(response.output)
        return response
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
