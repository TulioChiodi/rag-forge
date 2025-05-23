import re
import asyncio
import aiofiles
import subprocess
from pathlib import Path
from pypdf import PdfReader
from typing import Optional
from tempfile import NamedTemporaryFile
from src.logging_conf import logger

async def extract_text_from_pdf(path: Path | str) -> Optional[str]:
    """Extract text from a PDF file using PyPDF.
    
    Args:
        path: Path to the PDF file
        
    Returns:
        str: Extracted text if successful
        None: If no text could be extracted (triggers OCR fallback)
        
    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        ValueError: If the file is not a valid PDF
    """
    try:
        # Run PDF reading in a separate thread since it's blocking I/O
        reader = await asyncio.to_thread(PdfReader, path)

        # If first page contains text, extract text from all pages in the same thread pool
        if reader.pages[0].extract_text():
            def extract_all() -> str:
                return "\n".join(p.extract_text() for p in reader.pages)

            return await asyncio.to_thread(extract_all)

        # If no text found, signal OCR fallback
        return None  # trigger OCR fallback
    except Exception as e:
        logger.error(f"Error extracting text from PDF {path}: {str(e)}")
        raise

async def run_ocrmypdf(in_path: Path | str, out_txt_path: Path | str) -> str:
    """Run OCR on a PDF file and extract text.
    
    Args:
        in_path: Path to input PDF file
        out_txt_path: Path where the extracted text will be saved
        
    Returns:
        str: Extracted text from the PDF
        
    Raises:
        subprocess.CalledProcessError: If OCR process fails
        FileNotFoundError: If input file doesn't exist or output cannot be written
    """
    with NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
        try:
            # Run OCR process
            process = await asyncio.create_subprocess_exec(
                "ocrmypdf",
                str(in_path),
                temp_pdf.name,  # Use NamedTemporaryFile
                "--sidecar", str(out_txt_path),
                "--quiet",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, "ocrmypdf", stderr)
                
            # Read the output file asynchronously
            async with aiofiles.open(out_txt_path, mode='r') as f:
                return await f.read()
                
        except Exception as e:
            logger.error(f"Error running OCR on PDF {in_path}: {str(e)}")
            raise
        finally:
            # Clean up temp file
            try:
                Path(temp_pdf.name).unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Failed to cleanup temporary file {temp_pdf.name}: {str(e)}")

def normalize_text(text: str) -> str:
    """Normalize text by removing hyphenation and extra whitespace.
    
    Args:
        text: Input text to normalize
        
    Returns:
        str: Normalized text with consistent spacing and no hyphenation
        
    Raises:
        TypeError: If input is not a string
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string")
        
    text = text.replace("-\n", "")
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    
    return text.strip() 