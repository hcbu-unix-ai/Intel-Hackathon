#!/bin/bash

export HF_HOME=${HF_HOME:-'./data/hf'}

exec streamlit run main.py --server.port ${PORT:-8505} --server.address "0.0.0.0"
