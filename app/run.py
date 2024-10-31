import subprocess
import os
 
# Command to run the FastAPI app
fastapi_cmd = ["uvicorn", "rag_server:app", "--host", "0.0.0.0", "--port", "8044", "--reload"]
 
# Command to run the Streamlit app
streamlit_cmd = ["streamlit", "run", "rag_client.py", "--server.port", "8504", "--server.address", "0.0.0.0"]
 
 
if __name__ == "__main__":
    # Create subprocesses for both FastAPI and Streamlit
    fastapi_process = subprocess.Popen(fastapi_cmd)
    streamlit_process = subprocess.Popen(streamlit_cmd)
 
    # Wait for both processes to complete (this will keep them running)
    try:
        fastapi_process.wait()
        streamlit_process.wait()
    except KeyboardInterrupt:
        # If user interrupts the process, terminate both FastAPI and Streamlit
        fastapi_process.terminate()
        streamlit_process.terminate()