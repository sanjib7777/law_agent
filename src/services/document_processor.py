
import re
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentProcessor:
    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 300):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", ","]
        )


    def extract_parts_and_chapters(self, text: str):
        """
        Extract preamble, parts, and chapters from the document text.
        Returns (preamble, part_chapters, chapter_dict, None) in order of granularity.
        """
        # Extract preamble if present
        preamble_pattern = r"(Preamble:.*?)(?=Part\s*-\s*\d+|Chapter\s*-\s*\d+|$)"
        preamble_match = re.search(preamble_pattern, text, re.DOTALL)
        preamble = preamble_match.group(1).strip() if preamble_match else None

        # Extract parts
        part_pattern = r"Part\s*-\s*\d+\s*(.*?)(?=Part\s*-\s*\d+|\Z)"
        parts = re.findall(part_pattern, text, re.DOTALL)

        part_chapters = {}
        chapter_pattern = r"^(Chapter\s*-\s*\d+)(.*?)(?=\nChapter\s*-\s*\d+|\Z)"
        
        # If parts found, extract chapters within each part
        if parts:
            print(f"[DocumentProcessor] Extracted {len(parts)} parts.")
            for idx, part in enumerate(parts, start=1):
                chapters = re.findall(chapter_pattern, part, re.MULTILINE | re.DOTALL)
                print(f"[DocumentProcessor] Part-{idx}: Extracted {len(chapters)} chapters.")
                part_chapters[f"Part-{idx}"] = {chapter[0].replace(" ", "").replace("-", "-"): chapter[1].strip() for chapter in chapters}
            return preamble, part_chapters, None, None

        # If no parts, try to extract chapters from the whole text
        chapters = re.findall(chapter_pattern, text, re.MULTILINE | re.DOTALL)
        if chapters:
            print(f"[DocumentProcessor] No parts found. Extracted {len(chapters)} chapters from full text.")
            chapter_dict = {chapter[0].replace(" ", "").replace("-", "-"): chapter[1].strip() for chapter in chapters}
            return preamble, None, chapter_dict, None

        # If no chapters, fallback to splitting the whole text
        print("[DocumentProcessor] No parts or chapters found. Fallback to splitting full text.")
        return preamble, None, None, text.strip()


    def process_document(self, text: str, filename: str) -> List[Dict[str, Any]]:
        """
        Process a document and return chunks with metadata.
        Metadata includes filename, part, chapter, and chunk number.
        Fallback: if no parts, look for chapters; if no chapters, split whole text.
        """
        processed_chunks = []
        preamble, part_chapters, chapter_dict, fallback_text = self.extract_parts_and_chapters(text)

        # Handle preamble
        if preamble:
            preamble_chunks = self.splitter.split_text(preamble)
            for i, chunk in enumerate(preamble_chunks):
                processed_chunks.append({
                    "content": chunk,
                    "metadata": {
                        "filename": filename,
                        "part": None,
                        "chapter": "preamble",
                        "chunk_index": i
                    }
                })

        # Handle parts and chapters
        if part_chapters:
            for part, chapters in part_chapters.items():
                for chapter, content in chapters.items():
                    chapter_chunks = self.splitter.split_text(content)
                    for i, chunk in enumerate(chapter_chunks):
                        processed_chunks.append({
                            "content": chunk,
                            "metadata": {
                                "filename": filename,
                                "part": part,
                                "chapter": chapter,
                                "chunk_index": i
                            }
                        })
        elif chapter_dict:
            # Fallback: chapters only
            for chapter, content in chapter_dict.items():
                chapter_chunks = self.splitter.split_text(content)
                for i, chunk in enumerate(chapter_chunks):
                    processed_chunks.append({
                        "content": chunk,
                        "metadata": {
                            "filename": filename,
                            "part": None,
                            "chapter": chapter,
                            "chunk_index": i
                        }
                    })
        elif fallback_text:
            # Fallback: split whole text
            text_chunks = self.splitter.split_text(fallback_text)
            for i, chunk in enumerate(text_chunks):
                processed_chunks.append({
                    "content": chunk,
                    "metadata": {
                        "filename": filename,
                        "part": None,
                        "chapter": None,
                        "chunk_index": i
                    }
                })

        return processed_chunks