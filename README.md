# Equity-Analyst-Langchain-Project
This is only the foundation Of langchain Embeddings Process and Retrival With FAISS Vector Database

## Run the Streamlit app

1. Create a virtual environment and install dependencies:

```
pip install -r requirements.txt
```

2. Set your API key(s):

```
# PowerShell
# For Gemini
$Env:GEMINI_API_KEY = "YOUR_GEMINI_KEY"
# For Groq (if you choose Groq provider in the sidebar)
$Env:GROQ_API_KEY = "YOUR_GROQ_KEY"
```

3. Start the app:

```
streamlit run main.py
```

## Using different model providers

Select the provider in the sidebar:

- Groq (OSS): Works with open-source models hosted by Groq.
	- Examples: `llama3-8b-8192`, `llama3-70b-8192`, `mixtral-8x7b-32768`, `gemma2-9b-it`.
	- You can also enter a custom model id.
- Google Gemini: Use concrete model IDs like `gemini-1.0-pro`, `gemini-1.5-flash`, `gemini-1.5-flash-8b`, `gemini-1.5-pro`.

### Notes on Gemini model names

- Use concrete model IDs like `gemini-1.5-flash` or `gemini-1.5-pro`.
- Avoid the `-latest` aliases (e.g., `gemini-1.5-flash-latest`) as they may return `NotFound` with the current SDKs.

### Embeddings and retrieval

- Embeddings are created locally using `sentence-transformers/all-MiniLM-L6-v2` and stored in FAISS.

## Environment variables

- `GEMINI_API_KEY`: Google Generative AI key used when provider is Gemini.
- `GROQ_API_KEY`: Groq API key used when provider is Groq.

## Features

- Enter up to 3 article URLs, generate chunks, compute embeddings, and build a FAISS index.
- Ask a financial-analysis question; the app retrieves relevant chunks and generates an answer with the selected Gemini model.
	- Provider-agnostic: Works with either Groq or Gemini chat models.