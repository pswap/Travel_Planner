"""Pinecone vector store compatible with Pinecone 8.x and Python 3.14."""

from __future__ import annotations

import uuid
from typing import Any, Iterable, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


class PineconeVectorStore(VectorStore):
    """
    Pinecone vector store using the pinecone package (v5+).
    Use this when langchain-pinecone is unavailable (e.g. on Python 3.14).
    """

    def __init__(
        self,
        index: Any,
        embedding: Embeddings,
        text_key: str = "text",
        namespace: Optional[str] = None,
    ):
        self._index = index
        self._embedding = embedding
        self._text_key = text_key
        self._namespace = namespace

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self._embedding

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        namespace: Optional[str] = None,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> List[str]:
        texts = list(texts)
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        metadatas = metadatas or [{} for _ in texts]
        ns = namespace or self._namespace

        embeddings = self._embedding.embed_documents(texts)
        vectors = []
        for id_, text, metadata, embedding in zip(ids, texts, metadatas, embeddings):
            meta = dict(metadata)
            meta[self._text_key] = text
            vectors.append({"id": id_, "values": embedding, "metadata": meta})

        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            self._index.upsert(vectors=batch, namespace=ns, **kwargs)

        return ids

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        namespace: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        embedding = self._embedding.embed_query(query)
        return self.similarity_search_by_vector_with_score(
            embedding, k=k, filter=filter, namespace=namespace, **kwargs
        )

    def similarity_search_by_vector_with_score(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[dict] = None,
        namespace: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        ns = namespace or self._namespace
        response = self._index.query(
            vector=embedding,
            top_k=k,
            include_metadata=True,
            namespace=ns,
            filter=filter,
            **kwargs,
        )
        docs: List[Tuple[Document, float]] = []
        matches = getattr(response, "matches", None) or (
            response.get("matches", []) if isinstance(response, dict) else []
        )
        for match in matches:
            metadata = getattr(match, "metadata", None) or (
                match.get("metadata", {}) if isinstance(match, dict) else {}
            )
            metadata = dict(metadata) if metadata else {}
            text = metadata.pop(self._text_key, "")
            score = getattr(match, "score", None) or (
                match.get("score", 0.0) if isinstance(match, dict) else 0.0
            )
            docs.append((Document(page_content=text, metadata=metadata), float(score)))
        return docs

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        namespace: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        docs_scores = self.similarity_search_with_score(
            query, k=k, filter=filter, namespace=namespace, **kwargs
        )
        return [doc for doc, _ in docs_scores]

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> "PineconeVectorStore":
        """Create a PineconeVectorStore from texts. Requires index= in kwargs."""
        index = kwargs.pop("index", None)
        if index is None:
            raise ValueError("index must be provided in kwargs for PineconeVectorStore.from_texts")
        text_key = kwargs.pop("text_key", "text")
        namespace = kwargs.pop("namespace", None)
        store = cls(index=index, embedding=embedding, text_key=text_key, namespace=namespace)
        store.add_texts(texts, metadatas=metadatas, ids=ids, **kwargs)
        return store
