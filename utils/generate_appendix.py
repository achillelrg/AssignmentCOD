"""
Appendix Generator

This script aggregates all relevant source code files from the project into a single 
Markdown document, formatted for easy inclusion in a report Appendix.

Captured Directories:
- experiments/
- optimizer/
- utils/
- benchmarks/

Captured Files:
- run_ultimate.bat
- run_ultimate.sh
- requirements.txt

Usage:
    python generate_appendix.py
"""
import os

def generate_appendix(output_file="APPENDIX_CODE.md"):
    # configuration
    target_dirs = ["experiments", "optimizer", "utils", "benchmarks"]
    root_files = ["run_ultimate.bat", "run_ultimate.sh", "requirements.txt"]
    
    extensions = (".py", ".bat", ".sh", ".txt")
    base_dir = os.getcwd()
    
    print(f"Generating Code Appendix in {output_file}...")
    
    with open(output_file, "w", encoding="utf-8") as out:
        out.write("# Appendix D: Full Source Code\n\n")
        out.write("This appendix contains the complete source code for the project.\n\n")
        
        # 1. Root Files
        for fname in root_files:
            if os.path.exists(fname):
                info(out, fname, base_dir)
        
        # 2. Directories
        for d in target_dirs:
            # Walk strictly, sorted
            dir_path = os.path.join(base_dir, d)
            if not os.path.exists(dir_path):
                continue
                
            # Get all files in this dir (non-recursive to keep sections clean, or recursive?)
            # Let's do recursive but sorted
            for root, _, files in os.walk(dir_path):
                # Sort files for consistency
                files.sort()
                
                for file in files:
                    if file.endswith(extensions) and file != "__init__.py":
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, base_dir)
                        
                        # Skip if in __pycache__
                        if "__pycache__" in rel_path:
                            continue
                            
                        info(out, rel_path, base_dir)

    print(f"✅ Success! File created: {output_file}")

def info(out_handle, rel_path, base_dir):
    """Writes a file block to the markdown."""
    full_path = os.path.join(base_dir, rel_path)
    ext = os.path.splitext(rel_path)[1].lower()
    
    lang_map = {
        ".py": "python",
        ".sh": "bash",
        ".bat": "cmd",
        ".txt": "text"
    }
    lang = lang_map.get(ext, "")
    
    out_handle.write(f"## File: `{rel_path}`\n")
    out_handle.write(f"```{lang}\n")
    
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()
            out_handle.write(content)
    except Exception as e:
        out_handle.write(f"# Error reading file: {e}\n")
        
    out_handle.write("\n```\n\n")
    # Page break for PDF generation (optional, commonly \newpage in latex or similar)
    out_handle.write("---\n\n")

if __name__ == "__main__":
    generate_appendix()
