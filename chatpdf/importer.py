"""PDF importer — minimal first cut."""
import os
from pathlib import Path
import pdfplumber
from typing import Iterable

from .memory import MemoryService

# Constants for chunking
CHUNK_SIZE = 1000  # Target size in characters
CHUNK_OVERLAP = 100 # Number of chars to overlap between chunks
# Define common separators for recursive splitting, from largest to smallest semantic unit
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

def _recursive_split(text: str, chunk_size: int, chunk_overlap: int, separators: list[str]) -> list[str]:
    """Helper function to recursively split text."""
    final_chunks = []
    separator = separators[0]
    remaining_separators = separators[1:]

    # Try splitting by the current separator
    if separator:
        splits = text.split(separator)
    else:
        # If separator is empty string, split by character
        splits = list(text)

    current_chunk = ""
    for i, part in enumerate(splits):
        # Re-add the separator (except for the first part or if separator is empty)
        part_to_add = part
        if i > 0 and separator:
             part_to_add = separator + part

        # If adding this part exceeds chunk size (considering overlap for next chunk)
        if len(current_chunk) + len(part_to_add) > chunk_size:
            # If the current chunk is not empty, add it
            if len(current_chunk) > 0:
                final_chunks.append(current_chunk)
            
            # If the part itself is larger than chunk size, recursively split it
            if len(part_to_add) > chunk_size and remaining_separators:
                final_chunks.extend(_recursive_split(part_to_add, chunk_size, chunk_overlap, remaining_separators))
                current_chunk = "" # Reset chunk after recursive split adds its parts
            else:
                 # Start the next chunk with the current part (or its beginning if overlap needed)
                 current_chunk = part_to_add # Simple case, start next chunk
                 # This basic recursive split doesn't explicitly handle adding overlap *back* 
                 # to the previous chunk, but focuses on splitting down large pieces.
                 # A more complex implementation might handle overlap more precisely.

        else:
            # Otherwise, add the part to the current chunk
            current_chunk += part_to_add

    # Add the last remaining chunk if it exists
    if current_chunk:
        # If the last chunk is too big, try splitting it further
        if len(current_chunk) > chunk_size and remaining_separators:
             final_chunks.extend(_recursive_split(current_chunk, chunk_size, chunk_overlap, remaining_separators))
        else:
            final_chunks.append(current_chunk)
            
    # Filter out potential empty strings that might result from splitting
    return [chunk for chunk in final_chunks if chunk.strip()]


MAX_CHARS = 1000 # Retain for compatibility? Or remove if unused? Let's keep for now.

def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> Iterable[str]:
    """Recursively splits text into chunks respecting separators."""
    # Simple implementation: just call the recursive helper directly.
    # More robust versions might handle edge cases or different splitting strategies.
    if not text:
        return []
    return _recursive_split(text, size, overlap, SEPARATORS)

def import_pdf(path: Path, mem: MemoryService) -> int:
    """Parse *path*, add PDF & page-chunks to MemoryService, return pdf_id."""
    filename_str = str(path.resolve()) # Use absolute path for uniqueness
    title = path.stem

    # Check if PDF already exists
    existing_pdf_id = mem.get_pdf_id_by_filename(filename_str)
    if existing_pdf_id:
        print(f"PDF '{filename_str}' already exists (ID: {existing_pdf_id}). Re-importing...")
        # Delete old PDF record and associated chunks/embeddings
        mem.delete_pdf_by_id(existing_pdf_id)
        print("--- Finished deleting old data --- ")

    # Add new PDF record
    print(f"Adding new record for PDF: {filename_str}")
    pdf_id = mem.add_pdf(filename_str, title=title)
    if pdf_id is None:
         print(f"[bold red]Error: Failed to add PDF record for {filename_str} to database.[/]")
         return -1 # Indicate failure

    # Process and add chunks
    try:
        with pdfplumber.open(path) as pdf:
            print(f"Processing {len(pdf.pages)} pages for PDF ID {pdf_id}...")
            for page_no, page in enumerate(pdf.pages, start=1):
                raw = page.extract_text(x_tolerance=1, y_tolerance=3) or ""
                chunk_count = 0
                for chunk in _chunk_text(raw):
                    mem.add_chunk(pdf_id, page_no, chunk)
                    chunk_count += 1
                # print(f"  Page {page_no}: Added {chunk_count} chunks.") # Optional verbose logging
    except Exception as e:
        print(f"[bold red]Error processing PDF {filename_str}: {e}[/]")
        # Clean up the partial PDF entry if processing failed
        mem.delete_pdf_by_id(pdf_id)
        return -1 # Indicate failure

    print(f"Successfully processed and stored chunks for PDF ID {pdf_id}.")
    # TODO: summarisation call goes here
    return pdf_id