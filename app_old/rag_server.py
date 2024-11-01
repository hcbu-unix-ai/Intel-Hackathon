import os
import time
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
from fastapi import FastAPI
from huggingface_hub import login, whoami
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain import hub
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer, pipeline
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from optimum.intel.openvino import OVModelForCausalLM
import re
import warnings
from transformers import AutoModel


warnings.simplefilter("ignore")
load_dotenv(verbose=True)



# model_id = 'Llama-3.2-3B'
model_id='Llama-3.2-1B'
model_name        = os.getenv('MODEL_NAME')
model_precision   = os.getenv('MODEL_PRECISION', "FP16")
inference_device  = os.getenv('INFERENCE_DEVICE')
ov_config         = {"PERFORMANCE_HINT":"LATENCY", "NUM_STREAMS":"1", "CACHE_DIR":os.getenv('CACHE_DIR')}
num_max_tokens    = int(os.getenv('NUM_MAX_TOKENS', 350))
rag_chain_type    = os.getenv('RAG_CHAIN_TYPE')
vectorstore_dir   = os.getenv('VECTOR_DB_DIR')
embeddings_model  = os.getenv('MODEL_EMBEDDINGS')
chroma_path       = os.getenv('CHROMA_PATH')



model = AutoModel.from_pretrained(embeddings_model, trust_remote_code=True) 
embeddings = HuggingFaceEmbeddings(
    model_name = embeddings_model,
    model_kwargs = {'device':'cpu'},
    encode_kwargs = {'normalize_embeddings':True}
)


vectorstore_dir = f'{vectorstore_dir}'
vectorstore = Chroma(persist_directory=chroma_path, embedding_function=embeddings)
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt", api_url="https://api.hub.langchain.com")

custom_prompt_template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the query of the user.
If you don't know the answer, just say that you don't know. Keep the answer concise. Only answer to the query. **Please generate responses in point and subpoints.**
Context: {context}
Question: {question}
Answer:
"""

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=custom_prompt_template
)


print(f'** Vector store : {vectorstore_dir}')

ov_model_path = f'./{model_name}/{model_precision}'
tokenizer = AutoTokenizer.from_pretrained(ov_model_path, trust_remote_code=True)
print(ov_model_path)
model = OVModelForCausalLM.from_pretrained(model_id=ov_model_path, device=inference_device, ov_config=ov_config, cache_dir=os.getenv('CACHE_DIR'))
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=num_max_tokens,max_length=4096,     # Total length of prompt + response
    temperature=0.2,     # Adjust temperature for controlled creativity
    top_p=0.7 )
llm = HuggingFacePipeline(pipeline=pipe)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type=rag_chain_type,
    retriever=retriever,
    chain_type_kwargs={"prompt": custom_prompt}
)


def run_generation(text_user_en):
    ans = qa_chain.run(text_user_en)

    matches = re.findall(r"Answer:\s*(.*?)(?=\nContext:|\nQuestion:|$)", ans, re.DOTALL)

    complete_answers = [match.strip() for match in matches if len(match.strip()) > 20]
    if complete_answers:
        return max(complete_answers, key=len)
    return ans.strip()





app = FastAPI()

@app.get('/chatbot/{item_id}')
async def root(item_id:int, query:str=None):
    if query:
        try:
            stime = time.time()
            ans = run_generation(query)
            etime = time.time()
            wc = len(ans.split())  # simple word count
            process_time = etime - stime
            words_per_sec = wc / process_time
            return JSONResponse(content={
                'response': f'{ans} \r\n\r\nWord count: {wc}, Processing Time: {process_time:6.1f} sec, {words_per_sec:6.2} words/sec'
            })
        except Exception as e:
            return JSONResponse(content={"error": str(e)}, status_code=500)
    return JSONResponse(content={'response': ''})


