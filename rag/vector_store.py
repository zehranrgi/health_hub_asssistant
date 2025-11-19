"""
ChromaDB Vector Store Implementation for CVS HealthHub AI
Handles document storage, embedding generation, and semantic search
"""
import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()


class VectorStore:
    """Manages vector database operations with ChromaDB"""

    def __init__(
        self,
        collection_name: str = "healthhub_knowledge",
        persist_directory: str = "./data/chroma_db",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize vector store with ChromaDB

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist vector data
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
            embedding_model: Sentence transformer model for embeddings
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Initialize embedding function (local, free, fast)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> int:
        """
        Add documents to vector store

        Args:
            documents: List of document texts
            metadatas: Optional metadata for each document
            ids: Optional IDs for each document

        Returns:
            Number of chunks added
        """
        if not documents:
            return 0

        # Split documents into chunks
        all_chunks = []
        all_metadatas = []
        all_ids = []

        for idx, doc in enumerate(documents):
            chunks = self.text_splitter.split_text(doc)

            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)

                # Create metadata
                metadata = metadatas[idx].copy() if metadatas and idx < len(metadatas) else {}
                metadata["chunk_index"] = chunk_idx
                metadata["total_chunks"] = len(chunks)
                all_metadatas.append(metadata)

                # Create ID
                doc_id = ids[idx] if ids and idx < len(ids) else f"doc_{idx}"
                all_ids.append(f"{doc_id}_chunk_{chunk_idx}")

        # Add to ChromaDB (embeddings generated automatically)
        self.collection.add(
            documents=all_chunks,
            metadatas=all_metadatas,
            ids=all_ids
        )

        return len(all_chunks)

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic similarity search

        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Optional metadata filters

        Returns:
            List of matching documents with metadata and scores
        """
        # Search in ChromaDB (query embedding generated automatically)
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            where=filter_dict
        )

        # Format results
        formatted_results = []
        if results['documents'] and len(results['documents']) > 0:
            for idx in range(len(results['documents'][0])):
                formatted_results.append({
                    "content": results['documents'][0][idx],
                    "metadata": results['metadatas'][0][idx] if results['metadatas'] else {},
                    "distance": results['distances'][0][idx] if results['distances'] else None,
                    "id": results['ids'][0][idx]
                })

        return formatted_results

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "total_chunks": count,
            "persist_directory": self.persist_directory
        }

    def reset_collection(self):
        """Delete all documents from collection"""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )


def initialize_vector_store() -> VectorStore:
    """Initialize and return vector store instance"""
    # Use absolute path to ensure consistency across different working directories
    import pathlib
    base_dir = pathlib.Path(__file__).parent.parent
    persist_dir = base_dir / "data" / "chroma_db"

    return VectorStore(
        collection_name="healthhub_knowledge",
        persist_directory=str(persist_dir),
        chunk_size=1000,
        chunk_overlap=200,
        embedding_model="all-MiniLM-L6-v2"  # Free, local, 384-dim embeddings
    )


if __name__ == "__main__":
    # Test vector store
    vs = initialize_vector_store()
    print(f"Vector Store Stats: {vs.get_collection_stats()}")
