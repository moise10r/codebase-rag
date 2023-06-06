import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
from openai import OpenAI
from langchain.schema import Document
from git import Repo
import os
import shutil
import tempfile

# Page configuration
st.set_page_config(
    page_title="CodeBase RAGBot",
    page_icon="🤖",
    layout="wide"
)

# Initialize configurations
SUPPORTED_EXTENSIONS = {'.py', '.js', '.tsx', '.jsx', '.ipynb', '.java',
                       '.cpp', '.ts', '.go', '.rs', '.vue', '.swift', '.c', '.h'}

IGNORED_DIRS = {'node_modules', 'venv', 'env', 'dist', 'build', '.git',
                '__pycache__', '.next', '.vscode', 'vendor'}

# Initialize clients
@st.cache_resource
def init_clients():
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY")
    )
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    pinecone_index = pc.Index("codebase-rag")
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    return client, pinecone_index, model

client, pinecone_index, model = init_clients()

def trim_context(text, max_chars=2000):
    """Trim context to avoid token limit issues"""
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text

def clone_repository(repo_url):
    """Clones a GitHub repository to a temporary directory."""
    try:
        temp_dir = tempfile.mkdtemp()
        Repo.clone_from(repo_url, temp_dir)
        return temp_dir
    except Exception as e:
        raise Exception(f"Failed to clone repository: {str(e)}")

def get_file_content(file_path, repo_path):
    """Get content of a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        rel_path = os.path.relpath(file_path, repo_path)
        return {"name": rel_path, "content": trim_context(content)}
    except Exception:
        return None

def get_main_files_content(repo_path):
    """Get content of supported code files from the local repository."""
    files_content = []
    for root, _, files in os.walk(repo_path):
        if any(ignored_dir in root for ignored_dir in IGNORED_DIRS):
            continue
        for file in files:
            if os.path.splitext(file)[1] in SUPPORTED_EXTENSIONS:
                file_path = os.path.join(root, file)
                file_content = get_file_content(file_path, repo_path)
                if file_content:
                    files_content.append(file_content)
    return files_content

def process_repository(repo_url):
    """Process repository and store embeddings."""
    try:
        # Clone repository
        repo_path = clone_repository(repo_url)

        # Get file contents
        file_content = get_main_files_content(repo_path)

        # Create documents
        documents = []
        for file in file_content:
            doc = Document(
                page_content=trim_context(f"{file['name']}\n{file['content']}"),
                metadata={"source": file['name']}
            )
            documents.append(doc)

        # Store in Pinecone
        vectorstore = PineconeVectorStore.from_documents(
            documents=documents,
            embedding=HuggingFaceEmbeddings(),
            index_name="codebase-rag",
            namespace=repo_url
        )

        # Cleanup
        shutil.rmtree(repo_path)

        return True, "Repository processed successfully"
    except Exception as e:
        return False, str(e)

def perform_rag(query, repo_url):
    """Perform RAG query with token limit handling."""
    try:
        # Get query embedding
        query_embedding = model.encode(query)

        # Query Pinecone with reduced matches
        top_matches = pinecone_index.query(
            vector=query_embedding.tolist(),
            top_k=3,  # Reduced number of matches
            include_metadata=True,
            namespace=repo_url
        )

        # Get trimmed contexts
        contexts = []
        for match in top_matches['matches']:
            if 'text' in match['metadata']:
                context = trim_context(match['metadata']['text'])
                contexts.append(context)

        # Create compact augmented query
        augmented_query = f"""Code context:
{' '.join(contexts[:2])}

Question: {query}"""

        # Simplified system prompt
        system_prompt = "You are a helpful coding assistant. Provide clear, concise technical answers."

        # Get LLM response
        llm_response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Using smaller model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": augmented_query}
            ]
        )

        return True, llm_response.choices[0].message.content
    except Exception as e:
        return False, str(e)

# Streamlit UI
st.title("🤖 CodeBase RAGBot")
st.markdown("Chat with your codebase using AI")

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'repo_url' not in st.session_state:
    st.session_state.repo_url = ''
if 'repo_processed' not in st.session_state:
    st.session_state.repo_processed = False

# Sidebar for repository input
with st.sidebar:
    st.header("Repository Settings")
    repo_url = st.text_input("GitHub Repository URL", 
                            placeholder="https://github.com/username/repo",
                            key="repo_url_input")

    if repo_url and repo_url != st.session_state.repo_url:
        with st.spinner("Processing repository..."):
            success, message = process_repository(repo_url)
            if success:
                st.session_state.repo_url = repo_url
                st.session_state.repo_processed = True
                st.success(message)
            else:
                st.error(f"Error: {message}")
                st.session_state.repo_processed = False

# Main chat interface
if st.session_state.repo_processed:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about the codebase..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                success, response = perform_rag(prompt, st.session_state.repo_url)
                if success:
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    st.error(f"Error: {response}")
else:
    st.info("👈 Please enter a GitHub repository URL in the sidebar to start chatting.")

# Footer
st.markdown("---")
