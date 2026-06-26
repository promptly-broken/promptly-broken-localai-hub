#!/usr/bin/env python3
"""
LLM Wiki Agent - Local Obsidian Vault Manager

This demo shows a local agent that manages an Obsidian vault using Ollama LLM,
Faiss for semantic search, and Redis for caching. The agent processes queries by:
1. Searching existing notes with vector similarity
2. Generating responses with LLM
3. Updating the vault with new markdown content

Usage:
    python main.py

Requirements:
    pip install ollama sentence-transformers faiss-cpu redis python-markdown watchdog numpy
"""

import os
import time
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple
import ollama
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import redis
import markdown
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

# --- Configuration ---
OBSIDIAN_VAULT_PATH = "./obsidian_vault"
VECTOR_INDEX_PATH = "./vector_index.faiss"
REDIS_HOST = "localhost"
REDIS_PORT = 6379
MODEL_NAME = "codestral:latest"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Core Logic ---

class ObsidianVaultManager:
    """Manages reading/writing files in an Obsidian vault."""
    
    def __init__(self, vault_path: str):
        self.vault_path = Path(vault_path)
        if not self.vault_path.exists():
            self.vault_path.mkdir(parents=True, exist_ok=True)
            print(f"Created vault directory: {vault_path}")

    def list_markdown_files(self) -> List[Path]:
        """List all markdown files in the vault."""
        return list(self.vault_path.glob("*.md"))

    def read_file(self, file_path: Path) -> str:
        """Read content from a markdown file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return ""

    def write_file(self, file_name: str, content: str) -> Path:
        """Write content to a markdown file."""
        file_path = self.vault_path / file_name
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Wrote file: {file_path}")
            return file_path
        except Exception as e:
            print(f"Error writing {file_path}: {e}")
            return Path("")


class VectorIndexManager:
    """Manages vector embeddings and similarity search using FAISS."""
    
    def __init__(self, index_path: str, embedding_model_name: str):
        self.index_path = index_path
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.index = None
        self.documents = []
        self._load_or_create_index()

    def _load_or_create_index(self):
        """Load existing index or create a new one."""
        try:
            self.index = faiss.read_index(self.index_path)
            print("Loaded existing vector index")
        except Exception as e:
            print(f"Creating new vector index: {e}")
            self.index = faiss.IndexFlatIP(384)  # Dimension for all-MiniLM-L6-v2

    def add_documents(self, documents: List[Tuple[str, str]]):
        """Add documents to the vector index."""
        if not documents:
            return
        
        texts = [doc[1] for doc in documents]
        embeddings = self.embedding_model.encode(texts)
        
        # Normalize embeddings for cosine similarity
        embeddings = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        
        # Store document metadata
        self.documents.extend(documents)
        print(f"Added {len(documents)} documents to vector index")
        
        # Save index
        faiss.write_index(self.index, self.index_path)

    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, str, float]]:
        """Search for similar documents."""
        if self.index.ntotal == 0:
            return []
        
        query_embedding = self.embedding_model.encode([query], convert_to_tensor=False)
        query_embedding = np.array(query_embedding, dtype=np.float32)
        faiss.normalize_L2(query_embedding)
        
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.documents):
                title, content = self.documents[idx]
                results.append((title, content, distance))
        
        return results


class LLMWikiAgent:
    """Local agent that manages an Obsidian vault using Ollama."""
    
    def __init__(self, vault_path: str, redis_host: str = "localhost", redis_port: int = 6379):
        self.vault_manager = ObsidianVaultManager(vault_path)
        self.vector_index = VectorIndexManager(VECTOR_INDEX_PATH, EMBEDDING_MODEL)
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.model_name = MODEL_NAME
        
        # Initialize with existing vault content
        self._initialize_from_vault()

    def _initialize_from_vault(self):
        """Initialize vector index with existing vault content."""
        print("Initializing from vault content...")
        files = self.vault_manager.list_markdown_files()
        documents = []
        
        for file_path in files:
            content = self.vault_manager.read_file(file_path)
            if content.strip():
                title = file_path.stem
                documents.append((title, content))
                print(f"Loaded: {title}")
        
        self.vector_index.add_documents(documents)
        print("Vault initialization complete")

    def process_query(self, query: str) -> str:
        """Process a user query and return a response."""
        # Check cache first
        cache_key = f"query:{hashlib.md5(query.encode()).hexdigest()}"
        cached_result = self.redis_client.get(cache_key)
        if cached_result:
            print("Using cached result")
            print(f"{Fore.GREEN}LLM Response:\n{Fore.WHITE}{cached_result}\n")
            return cached_result
        
        # Search for relevant documents
        print(f"Searching for: {query}")
        search_results = self.vector_index.search(query, top_k=3)
        
        # Prepare context for LLM
        context = "\n\n".join([f"{title}: {content}" for title, content, _ in search_results])
        
        # Generate response with LLM
        prompt = f"""
You are an assistant helping manage a knowledge base. Answer the following query based on the provided context.

Context:
{context}

Query: {query}

Answer:"""
        
        try:
            print(f"{Fore.GREEN}LLM Response:\n{Fore.WHITE}", end="", flush=True)
            stream = ollama.generate(model=self.model_name, prompt=prompt, stream=True)
            answer = ""
            for chunk in stream:
                text = chunk['response']
                print(text, end="", flush=True)
                answer += text
            print("\n")
            
            # Cache the result
            self.redis_client.setex(cache_key, 3600, answer)  # Cache for 1 hour
            
            return answer
        except Exception as e:
            print(f"Error generating LLM response: {e}")
            return "Sorry, I encountered an error processing your query."

    def update_vault(self, title: str, content: str) -> Path:
        """Update the vault with new content."""
        file_name = f"{title.replace(' ', '_')}.md"
        return self.vault_manager.write_file(file_name, content)

    def handle_user_interaction(self, query: str):
        """Handle a user query and update the knowledge base if needed."""
        print(f"\n{Fore.CYAN}--- Processing Query: {query} ---")
        
        # Get LLM response
        response = self.process_query(query)
        
        # Create a new note based on the query and response
        note_title = f"Query_{hashlib.md5(query.encode()).hexdigest()[:8]}"
        note_content = f"# {query}\n\n{response}"
        
        # Update vault with new note
        file_path = self.update_vault(note_title, note_content)
        
        # Add to vector index
        if file_path.exists():
            self.vector_index.add_documents([(note_title, note_content)])
            print(f"Added new note to index: {file_path}")
        
        return response


class VaultMonitor(FileSystemEventHandler):
    """Monitors vault directory for changes."""
    
    def __init__(self, agent: LLMWikiAgent):
        self.agent = agent

    def on_modified(self, event):
        if event.is_directory:
            return
        
        if event.src_path.endswith('.md'):
            print(f"Detected change in: {event.src_path}")
            # Rebuild index when a file changes
            self.agent._initialize_from_vault()


def main():
    """Main demo function."""
    print(f"{Fore.MAGENTA}Starting LLM Wiki Agent Demo")
    
    # Create agent
    try:
        agent = LLMWikiAgent(OBSIDIAN_VAULT_PATH)
    except Exception as e:
        print(f"{Fore.RED}Failed to initialize agent: {e}")
        return
    
    # Simulate user queries
    test_queries = [
        "What is the purpose of this knowledge base?",
        "How does vector search work in this system?",
        "Explain how LLMs are used for note creation"
    ]
    
    print(f"\n{Fore.MAGENTA}--- Simulating User Queries ---")
    for query in test_queries:
        try:
            response = agent.handle_user_interaction(query)
            print(f"\n{Fore.YELLOW}--- Response Summary ---\n{Fore.WHITE}{response[:100]}...\n")
        except Exception as e:
            print(f"{Fore.RED}Error processing query '{query}': {e}")
            
    print(f"\n{Fore.GREEN}Demo completed. Check the vault directory for generated notes.")

if __name__ == "__main__":
    main()