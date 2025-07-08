# 🧠 PDF Chat with Gemini

Chat with any PDF using Google’s Gemini AI. This simple Gradio app lets you upload a PDF, process it, and ask questions in natural language.

## 🔧 How to Run

```bash
# 1. Install dependencies
pip install gradio langchain langchain-community langchain-google-genai faiss-cpu python-dotenv

# 2. Add your API key to a .env file
echo "GOOGLE_API_KEY=your_key_here" > .env

# 3. Run the app
python your_script.py
```

## 🛠️ What It Uses

- Google Generative AI (Gemini)
- LangChain (PDF loader, embeddings, retrieval, memory)
- FAISS (for vector storage)
- Gradio (chat UI)

## ⚡ Example Use

- Upload any PDF
- Ask questions like:  
  > "What’s the summary of this document?"  
  > "Who is the author?"  
  > "What are the key topics?"
