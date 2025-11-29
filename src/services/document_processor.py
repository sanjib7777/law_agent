import tiktoken
from typing import List

class DocumentProcessor:
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.chunk_size = 1000
        self.chunk_overlap = 200


    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks with overlap
        """
        tokens = self.tokenizer.encode(text)

        chunks = []
        start_idx = 0

        while start_idx < len(tokens):
            end_idx = start_idx + self.chunk_size
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)

            start_idx += self.chunk_size - self.chunk_overlap

            if start_idx >= len(tokens):
                break
        
        return chunks
    

    def process_document(self, text:str, filename: str) -> List[str]:
        """
        Process a document and return chunks with metadata
        """
        chunks = self.split_text(text)

        processed_chunks = []
        for i, chunk in enumerate(chunks):
            processed_chunks.append({
                "content": chunk,
                "metadata": {
                    "chunk_index": i,
                    "filename": filename,
                    "total_chunks": len(chunks)
                }
            })
        
        return processed_chunks