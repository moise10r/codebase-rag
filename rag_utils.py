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

# Initialize configurations
SUPPORTED_EXTENSIONS = {'.py', '.js', '.tsx', '.jsx', '.ipynb', '.java',
                       '.cpp', '.ts', '.go', '.rs', '.vue', '.swift', '.c', '.h'}

IGNORED_DIRS = {'node_modules', 'venv', 'env', 'dist', 'build', '.git',
                '__pycache__', '.next', '.vscode', 'vendor'}

# Initialize clients
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pc.Index("codebase-rag")

# Initialize the sentence transformer model
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

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
        return {"name": rel_path, "content": content}
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
        documents = [
            Document(
                page_content=f"{file['name']}\n{file['content']}",
                metadata={"source": file['name']}
            )
            for file in file_content
        ]

        # Store in Pinecone
        vectorstore = PineconeVectorStore.from_documents(
            documents=documents,
            embedding=HuggingFaceEmbeddings(),
            index_name="codebase-rag",
            namespace=repo_url
        )

        # Cleanup
        shutil.rmtree(repo_path)

        return {"status": "success", "message": "Repository processed successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def perform_rag(query, repo_url):
    """Perform RAG query."""
    try:
        # Get query embedding
        query_embedding = model.encode(query)

        # Query Pinecone
        top_matches = pinecone_index.query(
            vector=query_embedding.tolist(),
            top_k=5,
            include_metadata=True,
            namespace=repo_url
        )

        # Get contexts
        contexts = [item['metadata']['text'] for item in top_matches['matches']]

        # Create augmented query
        augmented_query = "<CONTEXT>\n" + "\n\n-------\n\n".join(contexts[:10]) + \
                         "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" + query

        # Get LLM response
        system_prompt = """You are a Senior Software Engineer. Answer questions about the codebase 
                         based on the provided context. Be concise and technical."""

        llm_response = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": augmented_query}
            ]
        )

        return {"status": "success", "response": llm_response.choices[0].message.content}
    except Exception as e:
        return {"status": "error", "message": str(e)}