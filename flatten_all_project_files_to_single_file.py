# flatten_project.py
import os
import sys

def flatten_project(root_dir: str, output_file: str):
    with open(output_file, 'w', encoding='utf-8') as out:
        # 1. Write a tree view
        out.write("Project structure:\n")
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # Exclude .git and __pycache__ directories
            dirnames[:] = [d for d in dirnames if d not in ('.git', '__pycache__')]
            level = dirpath.replace(root_dir, '').count(os.sep)
            indent = ' ' * 4 * level
            out.write(f"{indent}{os.path.basename(dirpath)}/\n")
            subindent = ' ' * 4 * (level + 1)
            for fname in filenames:
                out.write(f"{subindent}{fname}\n")
        out.write("\n\n")

        # 2. Write file contents
        for dirpath, dirnames, filenames in os.walk(root_dir):
            dirnames[:] = [d for d in dirnames if d not in ('.git', '__pycache__')]
            for fname in filenames:
                full_path = os.path.join(dirpath, fname)
                rel_path = os.path.relpath(full_path, root_dir)
                out.write(f"{'='*60}\n")
                out.write(f"File: {rel_path}\n")
                out.write(f"{'='*60}\n")
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    out.write(content)
                except Exception as e:
                    out.write(f"[Could not read file: {e}]\n")
                out.write("\n\n")

if __name__ == '__main__':
    # Set root_dir to the project root (where this script is located)
    root = os.path.dirname(os.path.abspath(__file__))
    output = os.path.join(root, 'project_flat.txt')
    flatten_project(root, output)
    print(f"Flattened project saved to: {output}")