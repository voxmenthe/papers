#!/bin/bash

# Check if --safe argument is provided
SAFE_MODE=false
if [[ "$1" == "--safe" ]]; then
    SAFE_MODE=true
fi

# Find all .md files in papers_ocr_results subdirectories and copy them
# while preserving their names but flattening the directory structure
find papers_ocr_results -type f -name "*.md" -exec sh -c '
    SAFE_MODE="$1"
    for file do
        # Get just the filename without the path
        filename=$(basename "$file")
        target="extracted_papers_markdown/$filename"
        
        if [ "$SAFE_MODE" = "true" ]; then
            # If in safe mode, append a counter for duplicate files
            counter=1
            while [ -f "$target" ]; do
                base="${filename%.*}"
                ext="${filename##*.}"
                target="extracted_papers_markdown/${base}_${counter}.${ext}"
                counter=$((counter + 1))
            done
        fi
        
        cp "$file" "$target"
        echo "Copied: $file -> $target"
    done
' sh "$SAFE_MODE" {} +

echo "Markdown files copy completed!" 