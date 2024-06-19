import os
from fastapi import FastAPI
import uvicorn
from dotenv import load_dotenv
from routers import chat
from routers.chat import APIRouter, chat_router, configs

# Retrieve the OpenAI API key from the environment variable
openai_api_key = os.getenv(configs.OPEN_API_KEY)

app = FastAPI()

app.include_router(chat_router)

if __name__== '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8080)
