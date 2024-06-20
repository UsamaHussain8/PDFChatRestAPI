# PDFChatAPI
Don't have time to read through all your book or any PDF document? Can't find time to read through the pages with a lot of details? Well, this API has got your back.

The API uses an LLM (commonly GPT 3.5 Turbo) to summarize the document according to your own queries. You first provide the API with your PDF document. Then once the model has been trained
on the document, you can query the document regarding any topic you want summarized. The prompt best practices apply here too, as a sidenote. 

This API uses Retrieval Augmented Generation (RAG) methodology to overcome the shortcomings inherent in LLMs (hallucinations, cut-off date beyond which LLM lacks knowledge of the newer data).

LangChain, which is an open-source library for developing generative AI applications, has been used in this API (Python version). It provides various utilities and APIs out-of-the-box
for parsing PDF documents (and various other ones, for that matter), API for storing chat history, and provides modules for using LLMs as an API.
The REST API has been created in FastAPI, an asynchronous, modern, fast framework provided in Python. The endpoints of the API will be discussed in detail below.

