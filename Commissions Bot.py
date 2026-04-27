import streamlit as st
import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever

# --- 1. SETUP & CONFIGURATION ---
st.title("💬 Global Payments Commission Co-Pilot")
st.write("Ask me about SAR calculations, MID activations, or Rooftop credits.")

# Securely load API Key (In production, use Streamlit Secrets)
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# --- 2. DATA INGESTION & VECTOR DATABASE ---
# This function runs once to read your policies and build the database
@st.cache_resource
def build_vector_store():
    # Load all text files from the 'policies' folder
    loader = DirectoryLoader('./policies', glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()

    # Split the documents into smaller, searchable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # Convert text chunks into embeddings and store them in ChromaDB
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store

vector_store = build_vector_store()

# --- 3. BUILD THE RAG PIPELINE ---
# Set up the LLM (temperature=0 keeps it factual and prevents hallucinations)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Define the strict system prompt for the chatbot
system_prompt = (
    "You are an expert sales commission analyst for Global Payments. "
    "Use the provided context to answer the user's question accurately. "
    "If the answer is not contained in the context, say 'I do not have that information in the current policy documents.' "
    "Never guess or make up commission numbers.\n\n"
    "CRITICAL MID LOGIC INSTRUCTIONS:\n"
    "If a user asks to check the status of a specific MID, look it up in the context. Apply the following rules based on its status:\n"
    "- If status is 'CreditApp': Tell the rep the account has not been fully boarded yet. They will not receive any SAR or MID credit.\n"
    "- If status is 'InstallApp': Tell the rep they will receive the SAR credit listed, but no MID credit yet.\n"
    "- If status is 'Installed': Tell the rep they will receive the SAR credit listed AND they qualify for the $500 MID credit.\n\n"
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Create the retrieval chain
advanced_retriever = MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(), llm=llm
)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(advanced_retriever, question_answer_chain)

# --- 4. STREAMLIT CHAT INTERFACE ---
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user input
if user_input := st.chat_input("E.g., When do I get my $500 MID credit?"):
    # Display user message
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Run the RAG
    response = rag_chain.invoke({"input": user_input})
    answer = response["answer"]

    # Escape dollar signs so Streamlit doesn't render them as math equations
    safe_answer = answer.replace("$", "\\$")

    # Display the AI response
    with st.chat_message("assistant"):
        st.markdown(safe_answer)
    st.session_state.messages.append({"role": "assistant", "content": safe_answer})