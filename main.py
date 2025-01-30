import streamlit as st 
from langchain_core.messages import HumanMessage, AIMessage
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

# Configuration de l'historique de chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Bonjour ! Je suis Saad, votre assistant virtuel. Comment puis-je vous aider ?")
    ]

# Configuration de la base vectorielle
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

def get_vector_store_from_url(url):
    try:
        loader = WebBaseLoader(url)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        documents_chunks = text_splitter.split_documents(documents)
        
        embeddings = OpenAIEmbeddings()
        vector_store = Chroma.from_documents(
            documents_chunks,
            embeddings,
            persist_directory="./chroma_db"
        )
        return vector_store
        
    except Exception as e:
        st.error(f"Erreur lors du traitement de l'URL : {str(e)}")
        return None

def get_context_retriever_chain():
    llm = ChatOpenAI()
    
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Générez une requête de recherche basée sur la conversation")
    ])
    
    retriever = st.session_state.vector_store.as_retriever()
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain

def get_conversational_rag_chain():
    llm = ChatOpenAI()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Répondez aux questions en français en utilisant ce contexte :\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    retriever_chain = get_context_retriever_chain()
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
    if not st.session_state.vector_store:
        return "Veuillez d'abord entrer une URL valide !"
    
    chain = get_conversational_rag_chain()
    
    response = chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    
    return response["answer"]

# Configuration de l'interface
st.set_page_config(
    page_title="Chat avec Sites Web",
    page_icon="💬",
    layout="wide"
)
st.title("Discutez avec n'importe quel site web 🌐")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    website_url = st.text_input("Entrez l'URL du site web:")
    
    if website_url and website_url != "":
        if st.button("Charger le contenu"):
            with st.spinner("Analyse du site web..."):
                vector_store = get_vector_store_from_url(website_url)
                if vector_store:
                    st.session_state.vector_store = vector_store
                    st.success("Contenu analysé avec succès !")

# Interaction principale
user_query = st.chat_input("Posez votre question ici...")
if user_query:
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.spinner("Réflexion..."):
        response = get_response(user_query)
        
    st.session_state.chat_history.append(AIMessage(content=response))

# Affichage de l'historique
for message in st.session_state.chat_history:
    role = "assistant" if isinstance(message, AIMessage) else "user"
    with st.chat_message(role):
        st.write(message.content)