import langchain
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
# from langchain.llms import OpenAI
from langchain.chains import ChatVectorDBChain
from langchain.chat_models import ChatOpenAI
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st

st.title("Chat with your Documents!")

OPENAI_API_KEY = st.secrets.OPENAI_API_KEY

chat_history = []
db = None

llm = ChatOpenAI(
        openai_api_key = OPENAI_API_KEY,
        model_name='gpt-3.5-turbo',
        temperature=0.0
    )

def pdf_loader(pdf_byte):
  reader = PyPDF2.PdfReader(pdf_byte)
  length = len(reader.pages)
  full_text = ""
  metadata_list = []

  for i in range(len(reader.pages)):
    page = reader.pages[i]
    text = text = page.extract_text()
    full_text = full_text + text
    metadata_list.append({"page number" : i})

  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size = 2000,
      chunk_overlap  = 100,
      length_function = len,
      add_start_index = True,
  )
  texts = text_splitter.create_documents([full_text])
  return texts
  

def loader(file_io):
    global db
    docs = pdf_loader(file_io)
    embed = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY
    )

    db = FAISS.from_documents(docs, embed)
    print("pdf loaded")
    return {"pdf":"loaded"}


def query_from_doc(text):
    global chat_history
    qa = ChatVectorDBChain.from_llm(llm, db)
    result = qa({"question": text, "chat_history": chat_history})
    print(result["answer"])
    chat_history = [(text, result["answer"])]
    return result["answer"]


if 1:
    uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf", "png", "jpg", "jpeg"])
    if uploaded_file:
        k = loader(uploaded_file)
        if k:
            question = user_input = st.text_input("Ask me anything!")
            st.write(query_from_doc(question))
        else:
            st.error("upload file!")
