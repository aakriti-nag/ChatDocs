import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Set page configuration
st.set_page_config(
    page_title="Chat with Multiple PDFs",
    page_icon="📄",
    layout="wide",
)

# Function to read and extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ''
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and save the FAISS vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local('faiss_index')

# Function to create the conversational chain with context
def get_conversational_chain():
    prompt_template = """
    Use the provided context and your own knowledge to answer the user's question as accurately as possible.
    Prioritize the context, but if additional information is needed, feel free to include it.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    chain = load_qa_chain(model, chain_type='stuff', prompt=prompt)
    return chain

# Function to create the chain without context
def get_chain_without_context():
    prompt_template = """
    The user has asked the following question. Please provide a detailed and accurate answer based on your knowledge.

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=['question'])
    chain = LLMChain(llm=model, prompt=prompt)
    return chain

# Function to handle user queries
def handle_user_input():
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    user_question = st.text_input('You:', key='input')
    if user_question:
        embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
        index_path = 'faiss_index/index.faiss'
        if not os.path.exists(index_path):
            st.error("FAISS index not found. Please upload and process PDF files first.")
            return

        with st.spinner('Generating response...'):
            vector_store = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)
            docs = vector_store.similarity_search(user_question, k=3)

            chain = get_conversational_chain()
            response = chain(
                {'input_documents': docs, 'question': user_question},
                return_only_outputs=True
            )

            answer = response['output_text']

            # Check if the answer indicates the information is not in the context
            if "Answer is not available in the context" in answer:
                st.info("The answer was not found in the PDFs. Trying to answer using the model's knowledge...")
                chain_no_context = get_chain_without_context()
                response = chain_no_context({'question': user_question}, return_only_outputs=True)
                answer = response['output_text']

            # Store the conversation
            st.session_state['chat_history'].append((user_question, answer))

        # Display conversation history with better formatting
        for question, answer in st.session_state['chat_history']:
            st.markdown(
                f"""
                <div style='background-color: #333333; padding: 15px; border-radius: 10px; margin-bottom: 10px;'>
                <b style='color: #00BFFF;'>You:</b> 
                <span style='color: #FFFFFF;'>{question}</span>
                </div>
                """, 
                unsafe_allow_html=True
                )
            st.markdown(
                f"""
                <div style='background-color: #444444; padding: 15px; border-radius: 10px; margin-bottom: 10px;'>
                <b style='color: #FFD700;'>AI:</b> 
                <span style='color: #FFFFFF;'>{answer}</span>
                </div>
                """, 
                unsafe_allow_html=True
                )

# Main function to run the app
def main():
    # Sidebar content
    with st.sidebar:
        st.title("📄 Chat with Your PDFs")
        st.write("Upload your PDF files and start asking questions!")

        pdf_docs = st.file_uploader('Upload PDF files', accept_multiple_files=True, type=['pdf'])

        if st.button('Process PDFs'):
            if pdf_docs:
                with st.spinner('Processing...'):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.session_state['faiss_index_created'] = True
                    st.success('Processing complete. You can now ask questions.')
            else:
                st.error("Please upload at least one PDF file.")

        st.write("#### Connect me on:")

        #Linkedin
        linkedIn_html = """<a href="https://www.linkedin.com/in/aakriti-nag/" target="blank">
        <img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" alt="LinkedIn" style="width:30px; height:30px; margin-right:10px;">LinkedIn</a>
        """
        # Email HTML
        email_html = """<a href="mailto:aakritinag04@gmail.com" target="blank">
        <img src="https://upload.wikimedia.org/wikipedia/commons/4/4e/Gmail_Icon.png" alt="Email" style="width:30px; height:30px; margin-right:10px;">Email</a>
        """
        st.markdown(linkedIn_html, unsafe_allow_html=True)
        st.markdown(email_html, unsafe_allow_html=True)

    # Main content area
    st.header("Chat with Your PDFs 📚")
    st.write("Ask questions about the content of your uploaded PDF documents.")

    if 'faiss_index_created' in st.session_state and st.session_state['faiss_index_created']:
        handle_user_input()
    else:
        st.info("Please upload and process PDF files to begin.")

if __name__ == '__main__':
    main()
