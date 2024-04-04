import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader 
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub

load_dotenv()

def get_pdf_content(pdf_docs):
    raw_text = ""
    for pdf_doc in pdf_docs:
        pdf_reader = PdfReader(pdf_doc)
        for page in pdf_reader.pages:
            raw_text += page.extract_text()
    return raw_text

def chunk_text(raw_text):
    splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    text_chunks = splitter.split_text(raw_text)
    return text_chunks

def get_vector_rep(text_chunks):
    # open ai is quicker but charges
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vector_rep = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_rep

def get_conversation_chain(vector_store):
    llm = ChatOpenAI()
    # llm =HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature": 0.5, "max_length": 512})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(), memory=memory)
    return conversation_chain

def handle_user_input(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    

def main():
    st.set_page_config(page_title="Chat with my NHS docs", page_icon="🧊")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with your PDFs! :books:")
    user_question = st.text_input("Ask a question about your docs:")
    if user_question:
        handle_user_input(user_question)

    with st.sidebar:
        st.subheader("Your PDFs")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", type="pdf", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing..."):
                if pdf_docs:
                    # Get raw text from PDFs
                    raw_text = get_pdf_content(pdf_docs)
                    # Chunk the text for embedding
                    text_chunks = chunk_text(raw_text)
                    # st.write(text_chunks)
                    # Create the vector representation of the text using openai embeddings
                    vector_store = get_vector_rep(text_chunks)
                    # create conversation chain - takes history and returns next element (need to maintain state via st.session_state because streamlit doesn't maintain sometimes)
                    st.session_state.conversation = get_conversation_chain(vector_store)
                else:
                    st.warning("Please upload some PDFs first.")

if __name__ == "__main__":
    main()