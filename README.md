# ChatDocs
### Chat with Your PDFs 📄
Interactively chat with the content of PDF documents using Google's Generative AI models. Upload one or more PDF files and ask questions about their content. The application leverages advanced language models to provide accurate and context-aware answers based on uploaded documents.

### Features
1. Upload Multiple PDFs: Easily upload one or more PDF documents.
2. Contextual Question Answering: Ask questions about the content of your PDFs and receive detailed answers.
3. Fallback to Model Knowledge: If the answer isn't found in the PDFs, the app will attempt to answer using the AI model's own knowledge.
4. Conversational Interface: Chat-like interface that keeps track of your questions and answers.
5. Easy to Use: Simple and intuitive Streamlit web interface.

### Demo
<img width="1435" alt="Screenshot 2024-10-08 at 9 56 57 PM" src="https://github.com/user-attachments/assets/874c42c3-4b56-4d04-aceb-5aaf7b7ad79c">

### Installation
#### Prerequisites
1. Python 3.7 or higher
2. Google API Key for Generative AI (PaLM API)

### Dependencies
The application relies on the following Python packages:
1. streamlit
2. PyPDF2
3. langchain
4. langchain-google-genai
5. google-generativeai
6. faiss-cpu
7. python-dotenv

These are specified in the requirements.txt file.
