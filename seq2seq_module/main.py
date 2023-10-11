# from fastapi import FastAPI
# from generate_response import generate_response

# app = FastAPI()


# @app.get("/")
# async def get_response(request: str):
#     return generate_response(request, '/home/alex/projects/ml/ml_final_project/seq2seq_module/models/model.pth', '/home/alex/projects/ml/ml_final_project/seq2seq_module/models/vocab.pt')


from fastapi import FastAPI
from generate_response import generate_response

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/message")
async def get_response(request: str):
    return {"response": generate_response(request, '/home/alex/projects/ml/ml_final_project/seq2seq_module/models/model.pth', '/home/alex/projects/ml/ml_final_project/seq2seq_module/models/vocab.pt')}
