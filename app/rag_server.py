import os
from dotenv import load_dotenv
from huggingface_hub import login, whoami
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, pipeline
from langchain.prompts import PromptTemplate
from optimum.intel.openvino import OVModelForCausalLM
import re
import warnings

warnings.simplefilter("ignore")
load_dotenv(verbose=True)

model_id          = os.getenv('MODEL_ID', 'unsloth/Llama-3.2-1B-Instruct')
inference_device  = os.getenv('INFERENCE_DEVICE', 'cpu')
num_max_tokens    = int(os.getenv('NUM_MAX_TOKENS', 1024))
rag_chain_type    = os.getenv('RAG_CHAIN_TYPE','stuff')
embeddings_model  = os.getenv('MODEL_EMBEDDINGS','sentence-transformers/all-MiniLM-L6-v2')
chroma_path       = os.getenv('CHROMA_PATH','./data/embedding')
cache_dir         = os.getenv('CACHE_DIR','./data/cache')
temperature       = os.getenv('TEMPERATURE',0.2)
top_p             = os.getenv('TOP_P',0.7)
ov_config         = {"PERFORMANCE_HINT":"LATENCY", "NUM_STREAMS":"1", "CACHE_DIR": cache_dir}


def load_embeddings_model():
    embeddings = HuggingFaceEmbeddings(
        model_name = embeddings_model,
        model_kwargs = {'device': inference_device},
        encode_kwargs = {'normalize_embeddings': False}
    )
    return embeddings

def load_llm_model():
    print("Loading Model.. : ",model_id)
    model = OVModelForCausalLM.from_pretrained(model_id=model_id, device=inference_device, ov_config=ov_config)
    return model

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return tokenizer

def run_query(query, model, tokenizer, embeddings):

    custom_prompt_template = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.

    Context: {context}
    
    
    Question: {question}
    
    Answer: 
    """

    prompt_template = """
    Use the following context as your learned knowledge, inside <context></context> XML tags.
    
    <context>
    {context}
    </context>

    When answer to user:
    - If you don't know, just say that you don't know.
    - If you don't know when you are not sure, ask for clarification.
    Avoid mentioning that you obtained the information from the context.
    And answer according to the language of the user's question.

    Given the context information, answer the query.
    Question: {question}

    Answer: 
    """

    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=custom_prompt_template
    )

    vectorstore = Chroma(persist_directory=chroma_path, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    pipe = pipeline("text-generation", model=model, device=inference_device, tokenizer=tokenizer, max_new_tokens=num_max_tokens)
    llm = HuggingFacePipeline(
        pipeline=pipe, 
        model_kwargs={"temperature": temperature, "top_p": top_p}
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=rag_chain_type,
        retriever=retriever,
        chain_type_kwargs={"prompt": custom_prompt}
    )
    
    result = qa_chain.run(query)
    
    #print("Full Text: ", result)
    matches = re.search(r'Answer:\s*([\s\S]*)', result)

    #print("Matched: ",matches.group(1))
    return matches.group(1)



