# ml_project

# Overview
This is a sentence-transformers model: It maps sentences & paragraphs to a 384 dimensional dense vector space and was designed for semantic search. It has been trained on 215M (question, answer) pairs from diverse sources. 

# API Discription
FastApi docs -  https://ml-project12345.herokuapp.com/docs

# Inputs
query: str  
docs: list 

# Outputs
return {"result": doc_score_pairs} 

# Methods
Get /Root - Return "Hello World"

Post /Predict - "query" get string. "docs" takes a list of strings for analysis similarity with "query". Function returns list, wich contains strings and its' results of analysis.

# Intended uses
Our model is intented to be used for semantic search: It encodes queries / questions and text paragraphs in a dense vector space. It finds relevant documents for the given passages.
Note that there is a limit of 512 word pieces: Text longer than that will be truncated. Further note that the model was just trained on input text up to 250 word pieces. It might not work well for longer text.

# Training procedure
The full training script is accessible in this current repository: train_script.py.

# Need to install
FastAPI — framework which used for creation fast HTTP API-servers with built-in validation serialization and asynchronyсо.
Unicorn — lightweight, multi-platform and multi-architecture processor emulator.
sentence-transformers — model of transformation sentances
