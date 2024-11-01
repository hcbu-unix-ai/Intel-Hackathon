import os
from dotenv import load_dotenv
import streamlit as st
import shutil
import rag_server
import generate_embeddings

# Load environment variables
load_dotenv(verbose=True)

docs_folder = os.getenv('DOCUMENT_DIR', './data/docs')  
embeddings_folder = os.getenv('CHROMA_PATH', './docs/embedding')

if not os.path.exists(docs_folder):
    os.makedirs(docs_folder)

@st.cache_resource(show_spinner="Loading Model...")
def load_model():
    if os.path.exists(embeddings_folder):
      shutil.rmtree(embeddings_folder)
    return rag_server.load_llm_model(), rag_server.load_tokenizer(), rag_server.load_embeddings_model()

@st.cache_resource(show_spinner="Encoding Documents...")
def encoding_docs(uploaded_files):
    generate_embeddings.generate_data_store()
    
    
# Set up Streamlit interface
st.title('OpenVINO Q&A Chatbot')
#st.markdown(f'QA Server: {server_url}:{server_port}')

model, tokenizer, embeddings = load_model()

# Initialize the session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'embeddings_updated' not in st.session_state:
    st.session_state.embeddings_updated = False

# Clear the `docs` and `docs_embedding` folders at the start of a session
if 'session_initialized' not in st.session_state:
    # This is the first time this session is being initialized, clear out the docs and docs_embedding folders  
    for file in os.listdir(docs_folder):
        file_path = os.path.join(docs_folder, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Clear the docs_embedding folder
    if os.path.exists(embeddings_folder):
        shutil.rmtree(embeddings_folder)
        os.makedirs(embeddings_folder)  # Recreate the directory to avoid issues during embedding generation

    st.session_state.session_initialized = True

# Sidebar for file upload functionality
with st.sidebar:
    st.title("Manage Knowledge Base")
    uploaded_files = st.file_uploader("Upload a PDF file to add to the knowledge base", type=["pdf"], accept_multiple_files=True)
    if len(uploaded_files) > 0:
        for uploaded_file in uploaded_files:
            # Save the uploaded file to the docs folder
            save_path = os.path.join(docs_folder, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
        # Run the generate_embeddings.py script to update embeddings
        try:
            #result = subprocess.run(["python", "generate_embeddings.py"], capture_output=True, text=True)
            encoding_docs(uploaded_files)
            st.success("Embeddings have been successfully updated.")
            st.session_state.embeddings_updated = True  # Allow chat interaction after embeddings are updated
        except Exception as e:
            st.error(f"An error occurred while generating embeddings: {str(e)}")
            st.session_state.embeddings_updated = False

# Display chat interface only if embeddings are updated
if st.session_state.embeddings_updated:
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    # Chat input only enabled after embeddings are successfully updated
    prompt = st.chat_input('Your input here.')
    response = ""
    if prompt:
        with st.chat_message('user'):
            st.markdown(prompt)
            st.session_state.messages.append({'role': 'user', 'content': prompt})
            try: 
                response = rag_server.run_query(prompt, model, tokenizer, embeddings)
                #print("Responce: ", response)
            except Exception as e:
                response = f"An error occurred: {str(e)}"

        with st.chat_message('assistant'):
            st.session_state.messages.append({'role': 'assistant', 'content': response})
            st.markdown(response)

else:
    # Show a message that embeddings need to be updated before chatting
    st.info("Upload a document to the knowledge base to start interacting with the chatbot.")
