import os
import shutil
from datetime import datetime
from dotenv import dotenv_values
from collections import namedtuple
import uuid
from fastapi import (File, UploadFile, APIRouter, Form, status)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_mongodb import MongoDBChatMessageHistory
import chromadb
from chromadb.utils import embedding_functions

config = dotenv_values(".env")
chat_router = APIRouter()
 
Constants = namedtuple('Constants', ['OPEN_API_KEY', 'EMBEDDINGS_MODEL', 'CHAT_MODEL', 'MONGO_CONNECTION_STRING', 'DB_PATH'])
configs = Constants(config["OPENAI_API_KEY"], config["EMBEDDINGS_MODEL"], config["CHAT_MODEL"], config['MONGO_CONNECTION_STRING'], config['DB_PATH'])

@chat_router.get("/getdocs/", status_code=status.HTTP_200_OK)
async def get_all_docs():
    client = chromadb.PersistentClient(path=configs.DB_PATH)
    collection = client.get_or_create_collection("PDF_Embeddings")
    vectordb = Chroma(persist_directory=configs.DB_PATH, collection_name = collection.name, client = client)

    doc_metadatas: list = vectordb.get(include=['metadatas'])['metadatas']
    results = [doc['source'].split("\\")[-1] for doc in doc_metadatas]

    if len(results) == 0:
        return {"code": 400, "response": "NO PDF FILES PROVIDED FOR GENERATING EMBEDDINGS YET."}
    
    return {"code": 200, "response": list(set(results))}

@chat_router.post("/trainpdf/", status_code=status.HTTP_201_CREATED)
async def create_upload_file(user_id: str = Form(...), pdf_file: UploadFile = File(...)):
    if not pdf_file.filename.endswith(".pdf"):
        return {"code": 400, "answer": "Only PDF files are allowed."}
    
    client = chromadb.PersistentClient(path=configs.DB_PATH)
    collection = client.get_or_create_collection("PDF_Embeddings")
    vectordb = Chroma(persist_directory=configs.DB_PATH, collection_name = collection.name, client = client)

    if check_for_existing_embeddings(pdf_file.filename, vectordb):
        return {"code": 400, "answer": "PDF EMBEDDINGS HAVE ALREADY BEEN GENERATED FOR THIS FILE. PLEASE PROVIDE A NEW FILE."}
    
    pdf_folder_path = f"Training_Data"
    os.makedirs(pdf_folder_path, exist_ok=True)
    
    file_path = os.path.join(pdf_folder_path, pdf_file.filename)
    with open(file_path, "wb") as temp_dest_file:
        temp_dest_file.write(await pdf_file.read())
        
    docs = read_docs(file_path, user_id)
    vectordb = generate_and_store_embeddings(docs, pdf_file, user_id)
    shutil.rmtree(pdf_folder_path, ignore_errors=True)

    if vectordb is None:
        return {"code": 400, "answer": "Error Occurred during Data Extraction from Pdf."}

    return {"code": 201, "answer": "PDF EMBEDDINGS GENERATED SUCCESSFULLY"}

@chat_router.post("/deletepdf/", status_code=status.HTTP_200_OK)
async def delete_pdf_doc(pdf_file: UploadFile = File(...)):
    embeddings = OpenAIEmbeddings(openai_api_key=configs.OPEN_API_KEY)
    client = chromadb.PersistentClient(path=configs.DB_PATH)
    collection = client.get_or_create_collection("PDF_Embeddings", embedding_function=embedding_functions.OpenAIEmbeddingFunction(api_key=config["OPENAI_API_KEY"], model_name=configs.EMBEDDINGS_MODEL))
    vectordb = Chroma(persist_directory=configs.DB_PATH, embedding_function=embeddings, collection_name = collection.name)

    doc_metadatas: list = vectordb.get(include=['metadatas'])['metadatas']
    for doc in doc_metadatas:
        if pdf_file.filename in doc['source']:
            doc['source'] = pdf_file.filename 
    data_associated_with_ids = vectordb.get(where={"source": pdf_file.filename})
    
    if len(data_associated_with_ids["ids"]) != 0:
        vectordb.delete(data_associated_with_ids["ids"])
        return {"code": 200, "answer": "PDF EMBEDDINGS DELETED SUCCESSFULLY"}
    
    return {"code": 400, "answer": "PDF EMBEDDINGS NOT FOUND FOR THIS FILE. PLEASE PROVIDE THE FILE FOR GENERATING."}

@chat_router.post("/deletepdfbyname/", status_code=status.HTTP_200_OK)
async def delete_pdf_doc_by_name(pdf_file_name: str):
    embeddings = OpenAIEmbeddings(openai_api_key=configs.OPEN_API_KEY)
    client = chromadb.PersistentClient(path=configs.DB_PATH)
    collection = client.get_or_create_collection("PDF_Embeddings", embedding_function=embedding_functions.OpenAIEmbeddingFunction(api_key=config["OPENAI_API_KEY"], model_name=configs.EMBEDDINGS_MODEL))
    vectordb = Chroma(persist_directory=configs.DB_PATH, embedding_function=embeddings, collection_name = collection.name)

    data_associated_with_ids = vectordb.get(where={"source": pdf_file_name}) 
 
    if len(data_associated_with_ids["ids"]) == 0:
        return {"code": 400, "answer": "PDF EMBEDDINGS NOT FOUND FOR THIS FILE. PLEASE PROVIDE THE FILE FOR GENERATING."}
    
    vectordb.delete(data_associated_with_ids["ids"])
    return {"code": 200, "answer": "PDF EMBEDDINGS DELETED SUCCESSFULLY"}  

@chat_router.post("/chatpdf/", status_code=status.HTTP_200_OK)
async def pdf_chat(query_params: dict):
    user_id: str = query_params.get('user_id')
    query: str = query_params.get('query')
    session_id: str = user_id + "-" + datetime.now().strftime("%d/%m/%Y")

    embeddings = OpenAIEmbeddings(openai_api_key=configs.OPEN_API_KEY)
    #client = chromadb.Client(Settings(persist_directory="./trained_db"))
    client = chromadb.PersistentClient(path=configs.DB_PATH)
    collection = client.get_or_create_collection("PDF_Embeddings", embedding_function=embedding_functions.OpenAIEmbeddingFunction(api_key=config["OPENAI_API_KEY"], model_name=configs.EMBEDDINGS_MODEL))
    vectordb = Chroma(persist_directory=configs.DB_PATH, embedding_function=embeddings, collection_name = collection.name)
    
    """Retrieve the documents relevant to the query and generate the response."""
    retriever = vectordb.as_retriever(search_type="mmr")
    relevant_docs = retriever.get_relevant_documents(query)
        
    """Now I am going about adding chat history into two ways. Both have their share of problems.
       1. Adding chat history to the prompt template. This method takes in chat history as context. But it returns the error:
          ValueError: Missing some input keys: {'context'}
          Note that the error is returned once the user asks a second question after the chat model responds to the first one.
    """
    prompt_template = """You are engaged in conversation with a human,
                          your responses will be generated using a comprehensive long document as a contextual reference. 
                          You can summarize long documents and also provide comprehensive answers, depending on what the user has asked.
                          You also take context in consideration and answer based on chat history.
                          Chat History: {chat_history}

                          Question: {question}

                          Answer :
                        """
    
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["chat_history", "question"])

    model = configs.CHAT_MODEL
    streaming_llm = ChatOpenAI(openai_api_key=configs.OPEN_API_KEY, model = model, temperature = 0.1, streaming=True)

    # use the streaming LLM to create a question answering chain
    # qa_chain = load_qa_chain(
    #     llm=streaming_llm,
    #     chain_type="stuff",
    #     prompt=PROMPT
    # )
    # question_generator_chain = LLMChain(llm=streaming_llm, prompt=PROMPT)
    # qa_chain_with_history = ConversationalRetrievalChain(
    #     retriever = vectordb.as_retriever(search_kwargs={'k': 3}, search_type='mmr'),
    #     combine_docs_chain=qa_chain,
    #     question_generator=question_generator_chain
    # )
    # response = qa_chain_with_history(
    #     {"question": query, "chat_history": user_specific_chat_memory.messages}
    # )

    # user_specific_chat_memory.add_user_message(response["question"])
    # user_specific_chat_memory.add_ai_message(response["answer"])

    """2. Adding chat history to the memory. This saves the memory in a buffer which is passed to the retrieval chain. 
    But it forgets the entire context of the conversation once the session restarts (even though messages are being added to MongoDB).
    """
    mongodb_chat_client = MongoDBChatMessageHistory(
        connection_string=configs.MONGO_CONNECTION_STRING, 
        session_id=session_id, 
        collection_name="Chat_History",
        #create_index = True
        )
    memory = ConversationBufferMemory(
    memory_key="chat_history",
    chat_memory=mongodb_chat_client,
    output_key="answer",
    return_messages=True
)
    
    qa_chain_with_history = ConversationalRetrievalChain.from_llm(
    ChatOpenAI(openai_api_key = config["OPENAI_API_KEY"], model_name = model, temperature = 0.1),
    retriever = vectordb.as_retriever(search_kwargs={'k': 3}, search_type='mmr'),
    memory = memory,
    chain_type="stuff"
)
    result = qa_chain_with_history.invoke({'question': query})

    return {"code": 200, "answer": result["answer"]}
    
def read_docs(pdf_file, user_id: str):
    pdf_loader = PyPDFLoader(pdf_file)
    pdf_documents = pdf_loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(pdf_documents)
    
    now = datetime.now()
    for doc in documents:
        doc.metadata = {
            "user": user_id,
            "id": str(uuid.uuid4()),  
            "source": pdf_file.split("\\")[-1],
            'created_at': now.strftime("%d/%m/%Y %H:%M:%S")
        }

    return documents

def generate_and_store_embeddings(documents, pdf_file, user_id):
    client = chromadb.PersistentClient(path=configs.DB_PATH)
    collection = client.get_or_create_collection("PDF_Embeddings",
                                                 embedding_function=embedding_functions.OpenAIEmbeddingFunction(api_key=config["OPENAI_API_KEY"],
                                                                                                                model_name=configs.EMBEDDINGS_MODEL))
    
    try:
        vectordb = Chroma.from_documents(
                    documents,
                    embedding=OpenAIEmbeddings(openai_api_key=config["OPENAI_API_KEY"], model=configs.EMBEDDINGS_MODEL),
                    persist_directory=configs.DB_PATH,
                    collection_name = collection.name, 
                    client = client
                )
        vectordb.persist()
        
    except Exception as err:
        return None
    
    return vectordb

def check_for_existing_embeddings(pdf_filename, vectordb):
    doc_metadatas: list = vectordb.get(include=['metadatas'])['metadatas']
    results = [doc['source'].split("\\")[-1] for doc in doc_metadatas]
    if pdf_filename in list(set(results)):
        return True
    

def get_message_history(session_id: str) -> MongoDBChatMessageHistory:
    return MongoDBChatMessageHistory(connection_string=configs.MONGO_CONNECTION_STRING, 
                                     session_id=session_id, 
                                     collection_name="Chat_History")