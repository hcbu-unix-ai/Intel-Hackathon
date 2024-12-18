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

model_id          = os.getenv('MODEL_ID', 'unsloth/Llama-3.2-3B-Instruct')
inference_device  = os.getenv('INFERENCE_DEVICE', 'cpu')
num_max_tokens    = int(os.getenv('NUM_MAX_TOKENS', 1024))
rag_chain_type    = os.getenv('RAG_CHAIN_TYPE','stuff')
embeddings_model  = os.getenv('MODEL_EMBEDDINGS','sentence-transformers/all-MiniLM-L6-v2')
chroma_path       = os.getenv('CHROMA_PATH','./data/embedding')
cache_dir         = os.getenv('CACHE_DIR','./data/cache')
temperature       = os.getenv('TEMPERATURE',0.2)
top_p             = os.getenv('TOP_P',0.6)
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
    model = OVModelForCausalLM.from_pretrained(model_id=model_id, device=inference_device, ov_config=ov_config, export=True)
    return model

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return tokenizer


def clean_context(context):

    context = re.sub(r'\n+', '\n', context) 
    context = re.sub(r'\s+', ' ', context)
    context = context.strip()
    return context


def run_query(query, model, tokenizer, embeddings):

    prompt_template_text = """
    You are an assistant for answering questions concisely based on the provided context. 
    Only answer to the query in a concise manner, focusing only on the main points. 

    Context:
    {context}
        
    Question: {question}

    Answer:
    """

    prompt_template = """
Use the following context as your learned knowledge, inside <context></context> XML tags.

<context>
{context}
</context>

When answer to user:
- Always respond to the greetings.
- If you don't know, just say that you don't know.
- If you don't know when you are not sure, ask for clarification.
Avoid mentioning that you obtained the information from the context.
And answer according to the language of the user's question.

Given the context information, answer the query.
query: {question}
"""

    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template_text
    )

    vectorstore = Chroma(persist_directory=chroma_path, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    retrieved_docs = retriever.get_relevant_documents(query)
    cleaned_contexts = []
    seen_texts = set()

    for doc in retrieved_docs:
        cleaned_content = clean_context(doc.page_content)
        if cleaned_content and cleaned_content not in seen_texts:
            seen_texts.add(cleaned_content)
            cleaned_contexts.append(cleaned_content)

    combined_context = "\n".join(cleaned_contexts)
    qa_prompt = custom_prompt.format(context=combined_context, question=query)

    pipe = pipeline("text-generation", model=model, device=inference_device, tokenizer=tokenizer, max_new_tokens=num_max_tokens)
    llm = HuggingFacePipeline(
        pipeline=pipe, 
        model_kwargs={"temperature": temperature, "top_p": top_p}
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=rag_chain_type,
        retriever=retriever,
        # chain_type_kwargs={"prompt": custom_prompt}
    )
    
    result = qa_chain.run(qa_prompt)
    
    #print("Full Text: ", result)
    matches = re.search(r'Helpful Answer:\s*([\s\S]*)', result)

    #print("Matched: ",matches.group(1))
    return matches.group(1)

