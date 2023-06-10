# CodeBase RAGBot 

## What is CodeBase RAGBot?

CodeBase RAGBot is an innovative tool that transforms how developers understand and interact with codebases. By combining the power of Retrieval-Augmented Generation (RAG) with large language models, it creates an intuitive chat interface for exploring and understanding GitHub repositories. Simply provide a repository URL, and CodeBase RAGBot will analyze the codebase, understand its structure, and allow you to ask questions in natural language. Whether you're a new developer trying to understand a project, or an experienced programmer looking to quickly navigate complex codebases, CodeBase RAGBot acts as your intelligent coding companion, providing context-aware answers and insights about the code.

Unlike traditional code search tools, CodeBase RAGBot understands the context and relationships within the code, allowing it to provide more meaningful and comprehensive answers. It can explain code functionality, suggest improvements, identify patterns, and help you understand how different parts of the codebase interact with each other. This makes it an invaluable tool for code review, onboarding, and maintaining large projects.

## Features 

- **Repository Analysis**: Load and analyze any public GitHub repository
- **Smart Code Understanding**: Uses RAG to provide context-aware responses about the codebase
- **Interactive Chat**: Natural conversation interface to ask questions about the code
- **Multiple Language Support**: Supports Python, JavaScript, TypeScript, Java, and more
- **Token-Optimized**: Efficient handling of large codebases with smart context management

## Tech Stack 

- **Frontend**: Streamlit
- **Embeddings**: Sentence Transformers
- **Vector Store**: Pinecone
- **LLM**: Groq (llama-3.1-70b-versatile)
- **Repository Management**: GitPython

## Getting Started 

### Prerequisites
- Python 3.8+
- Pinecone API Key
- Groq API Key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/aneezahere/codebase-rag.git
cd codebase-rag
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export GROQ_API_KEY='your_groq_api_key'
export PINECONE_API_KEY='your_pinecone_api_key'
```

4. Run the application:
```bash
streamlit run main.py
```

## Usage 

1. Enter a GitHub repository URL
2. Wait for the repository to be processed
3. Start asking questions about the codebase
4. Get AI powered responses based on the actual code

## Live Demo 🌐

Try out the live demo: https://codebase-rag.replit.app

## Authors

👤 NGANULO RUSHANIKA Moise

- GitHub: [@githubhandle](https://github.com/moise10r)
- Twitter: [@twitterhandle](https://twitter.com/MRushanika)
- LinkedIn: [LinkedIn](https://www.linkedin.com/in/nganulo-rushanika-mo%C3%AFse-626139197/)
## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

Feel free to check the [issues page](../../issues/).

## Show your support

Give a ⭐️ if you like this project!

## Acknowledgments 

- Built with [Streamlit](https://streamlit.io/)
- Powered by [Groq](https://groq.com/)
- Vector search by [Pinecone](https://www.pinecone.io/)

