import os
import pdfplumber
from docx import Document
from io import BytesIO
from fastapi import HTTPException, UploadFile
import logging


logger = logging.getLogger(__name__)

class FileParser:
    """Parser for different file types"""
    
    @staticmethod
    async def parse_file(file: UploadFile, filename: str) -> str:
        """Parse file content based on file extension"""
        file_extension = os.path.splitext(filename)[1].lower()
        
        try:
            if file_extension == '.pdf':
                return await FileParser._parse_pdf(file)
            elif file_extension == '.docx':
                return await FileParser._parse_docx(file)
            elif file_extension in ['.txt', '.md', '.csv']:
                return await FileParser._parse_text(file)
            else:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Unsupported file type: {file_extension}. Supported types: PDF, DOCX, TXT, MD, CSV"
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error parsing file {filename}: {str(e)}")
            raise HTTPException(
                status_code=400, 
                detail=f"Failed to parse file {filename}: {str(e)}"
            )
    
    @staticmethod
    async def _parse_pdf(file: UploadFile) -> str:
        """Parse PDF file content"""
        text = ""
        try:

            file_content = await file.read()

            # Wrap the byte content in a file-like object
            pdf_file = BytesIO(file_content)

            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
            
            if not text.strip():
                raise HTTPException(
                    status_code=400, 
                    detail="PDF appears to be empty or contains no extractable text"
                )
            
            return text
            
        except Exception as e:
            logger.error(f"PDF parsing error: {str(e)}")
            raise HTTPException(
                status_code=400, 
                detail=f"Failed to parse PDF file: {str(e)}"
            )
    
    @staticmethod
    async def _parse_docx(file: UploadFile) -> str:
        """Parse DOCX file content"""
        try:
            doc = Document(file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            
            if not text.strip():
                raise HTTPException(
                    status_code=400, 
                    detail="DOCX file appears to be empty"
                )
            
            return text
            
        except Exception as e:
            logger.error(f"DOCX parsing error: {str(e)}")
            raise HTTPException(
                status_code=400, 
                detail=f"Failed to parse DOCX file: {str(e)}"
            )
    
    @staticmethod
    async def _parse_text(file: UploadFile) -> str:
        """Parse text-based files (TXT, MD, CSV)"""
        try:
            # Try UTF-8 first
            try:
                return file.decode('utf-8')
            except UnicodeDecodeError:
                # Fallback to latin-1 if UTF-8 fails
                return file.decode('latin-1')
        except Exception as e:
            logger.error(f"Text file parsing error: {str(e)}")
            raise HTTPException(
                status_code=400, 
                detail=f"Failed to parse text file: {str(e)}"
            )
    
    @staticmethod
    def get_supported_formats() -> list:
        """Return list of supported file formats"""
        return ['.pdf', '.docx', '.txt', '.md', '.csv']