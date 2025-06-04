import os
import warnings
import logging
import streamlit as st

from langdetect import detect
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA

# Disable warnings and unnecessary logs
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# ✅ HARDCODED API KEY (your choice)
#GROQ_API_KEY = "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"
GROQ_API_KEY = "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"

# Streamlit UI setup
st.set_page_config(page_title="RegioWizard KI", layout="centered")
st.title('🧠 RegioWizard KI')

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# Greeting detection
def is_greeting(text):
    return text.lower().strip() in ["hi", "hello", "hey", "greetings", "hallo", "servus", "moin"]

# Language detection
def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

# Cached vectorstore creation
@st.cache_resource
def get_vectorstore():
    pdf_path = os.path.join(os.path.dirname(__file__), "bad_breisig_docs.pdf")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")
    loaders = [PyPDFLoader(pdf_path)]
    return VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2'),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    ).from_loaders(loaders).vectorstore

# Get user input
prompt = st.chat_input('Pass your prompt here')

if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    try:
        # Initialize the LLM
        groq_chat = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="llama3-8b-8192"
        )

        lang = detect_language(prompt)

        if is_greeting(prompt):
            response = (
                "Hallo, ich bin der RegioWizard_KI Chatbot! 😊 Frag mich alles über Bad Breisig!"
                if lang == "de" else
                "Hi, I'm RegioWizard_KI Chatbot! 😊 Ask me anything about Bad Breisig!"
            )
        else:
            # Load vectorstore
            vectorstore = get_vectorstore()

            # Prompt template
            qa_prompt = ChatPromptTemplate.from_template(
                """
Du bist ein hilfsbereiter Assistent mit Wissen über Bad Breisig.
Verwende AUSSCHLIESSLICH den untenstehenden Kontext, um die Frage des Nutzers zu beantworten.

Sei präzise, sachlich und gut verständlich.
Wenn die Antwort nicht im Kontext enthalten ist, sage: „Nicht im Kontext gefunden.“

Kontext:
{context}

Frage: {question}

Antwort:
                """ if lang == "de" else
                """
You are a helpful assistant knowledgeable about Bad Breisig.
Use ONLY the context below to answer the user's question.

Be concise, factual, and human-readable.
If the answer is not in the context, say: "Not found in the context."

Context:
{context}

Question: {question}

Answer:
                """
            )

            # Retrieval QA chain
            chain = RetrievalQA.from_chain_type(
                llm=groq_chat,
                chain_type='stuff',
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                chain_type_kwargs={"prompt": qa_prompt},
                return_source_documents=True
            )

            result = chain({"query": prompt})
            response = result["result"].strip()

            # Fallback if vague
            if not response or "not found" in response.lower() or "nicht im kontext" in response.lower():
                fallback_docs = vectorstore.similarity_search_with_score(prompt, k=3)
                keyword_hits = list({doc.page_content.strip()[:300] for doc, _ in fallback_docs})

                if keyword_hits:
                    response = (
                        "Hier sind die relevantesten Informationen:\n\n" if lang == "de" else
                        "Here’s the most relevant information found:\n\n"
                    )
                    response += "\n\n".join(keyword_hits)
                else:
                    response = (
                        "Nicht im bereitgestellten Dokument gefunden." if lang == "de" else
                        "Not found in the provided document."
                    )

        # Display response
        st.chat_message('assistant').markdown(response)
        st.session_state.messages.append({'role': 'assistant', 'content': response})

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
