# PDFChatAPI
Don't have time to read through all your book or any PDF document? Can't find time to read through the pages with a lot of details? Well, this API has got your back.

The API uses an LLM (commonly GPT 3.5 Turbo) to summarize the document according to your own queries. You first provide the API with your PDF document. Then once the model has been trained
on the document, you can query the document regarding any topic you want summarized. The prompt best practices apply here too, as a sidenote. 

This API uses [Retrieval Augmented Generation (RAG)](https://aws.amazon.com/what-is/retrieval-augmented-generation/) methodology to overcome the shortcomings inherent in LLMs (hallucinations, cut-off date beyond which LLM lacks knowledge of the newer data).

[LangChain](https://python.langchain.com/v0.2/docs/introduction/), which is an open-source library for developing generative AI applications, has been used in this API (Python version). It provides various utilities and APIs out-of-the-box
for parsing PDF documents (and various other ones, for that matter), API for storing chat history, and provides modules for using LLMs as an API.
The REST API has been created in [FastAPI](https://fastapi.tiangolo.com/), an asynchronous, modern, fast Python framework for creating APIs.

## How to install
You can use the API yourself on your local system. 
1. [Clone the repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) on your system.
2. Install the requirements for running this API and powering the endpoints on your localhost using `pip install -r requirements.txt` This will install all the required modules 
and packages by itself. Its better to [create a Python virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) and install the packages in that environment.

You will need to create a .env file on your system in the project folder (containing the main.py script file). You will need to add these 5 fields in the file.
1. OPENAI_API_KEY (your API Key for interacting with the LLM provided by the vendor).
2. EMBEDDINGS_MODEL (the model you want to use for generating embeddings for text, default is text-embedding-ada-002
3. CHAT_MODEL (by default we used gpt-3.5-turbo)
4. MONGO_CONNECTION_STRING (for storing chat history, for using local client, it is simply: mongodb://localhost:27017)
5. DB_PATH (use: ./trained_db for storing in main project directory)

Once all the packages have been installed and the environment variables defined, navigate to the project directory, run the script using: `python main.py` and power up the local client
using the link provided in the terminal or command prompt window. Append /docs to the provided URL to find all the API endpoints.
