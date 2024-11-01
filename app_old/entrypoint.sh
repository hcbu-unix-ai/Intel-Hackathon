#!/bin/bash

# Start the FastAPI server in the background
uvicorn rag_server:app --host 0.0.0.0 --port 8033 --reload &

# Start the Streamlit app
streamlit run rag_client.py --server.port 8502 --server.headless true
