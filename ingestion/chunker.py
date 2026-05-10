from dataclasses import dataclass, field
from typing import List
import tiktoken
from config import cfg
import re

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    text: str
    chunk_index: int
    token_count: int
    metadata: dict = field(default_factory=dict)


class RecursiveChunker:
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.chunk_size = cfg.chunk_size
        self.overlap = cfg.chunk_overlap

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def _split_into_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s for s in sentences if s.strip()]

    def chunk_document(self, doc_id: str, title: str, text: str, metadata: dict = None) -> List[Chunk]:
        if metadata is None:
            metadata = {}

        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks: List[Chunk] = []
        current_sentences: List[str] = []
        current_tokens: int = 0
        chunk_index: int = 0

        for paragraph in paragraphs:
            sentences = self._split_into_sentences(paragraph)

            for sentence in sentences:
                sentence_tokens = self._count_tokens(sentence)

                if current_tokens + sentence_tokens > self.chunk_size and current_sentences:
                    chunk_text = " ".join(current_sentences)
                    chunks.append(Chunk(
                        chunk_id=f"{doc_id}_chunk_{chunk_index:03d}",
                        doc_id=doc_id,
                        doc_title=title,
                        text=chunk_text,
                        chunk_index=chunk_index,
                        token_count=self._count_tokens(chunk_text),
                        metadata=metadata,
                    ))
                    chunk_index += 1

                    overlap_sentences: List[str] = []
                    overlap_tokens: int = 0
                    for s in reversed(current_sentences):
                        s_tokens = self._count_tokens(s)
                        if overlap_tokens + s_tokens > self.overlap:
                            break
                        overlap_sentences.insert(0, s)
                        overlap_tokens += s_tokens

                    current_sentences = overlap_sentences
                    current_tokens = overlap_tokens

                current_sentences.append(sentence)
                current_tokens += sentence_tokens

        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append(Chunk(
                chunk_id=f"{doc_id}_chunk_{chunk_index:03d}",
                doc_id=doc_id,
                doc_title=title,
                text=f"[{title}] {chunk_text}",
                chunk_index=chunk_index,
                token_count=self._count_tokens(chunk_text),
                metadata=metadata,
            ))

        return chunks