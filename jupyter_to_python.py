import json
import argparse
import os
from pathlib import Path


def convert_notebook_to_script(notebook_path, output_path=None, include_markdown=True):
    if output_path is None:
        output_path = str(Path(notebook_path).with_suffix('.py'))
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    script_content = [
        "# Converted from Jupyter notebook using notebook-to-python converter",
        f"# Original notebook: {os.path.basename(notebook_path)}",
        ""
    ]
    
    for i, cell in enumerate(notebook['cells']):
        cell_type = cell['cell_type']
        
        if cell_type == 'code':
            if i > 0:
                script_content.append("")
            
            script_content.extend([line.rstrip() for line in cell['source']])
        
        elif cell_type == 'markdown' and include_markdown:
            if i > 0:
                script_content.append("")
            
            script_content.append("# " + "=" * 60)
            script_content.append("# MARKDOWN CELL")
            script_content.append("# " + "=" * 60)
            for line in cell['source']:
                if line.strip():
                    script_content.append("# " + line.rstrip())
                else:
                    script_content.append("#")
            script_content.append("# " + "=" * 60)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(script_content))
    
    return output_path


def main():
    """Command line entry point."""
    parser = argparse.ArgumentParser(description='Convert Jupyter notebook to Python script')
    parser.add_argument('notebook', help='Path to the Jupyter notebook (.ipynb file)')
    parser.add_argument('--output', '-o', help='Output path for the Python script')
    parser.add_argument('--no-markdown', action='store_true', 
                        help='Skip markdown cells in the output')
    
    args = parser.parse_args()
    
    output_path = convert_notebook_to_script(
        args.notebook, 
        args.output, 
        not args.no_markdown
    )
    
    print(f"Converted {args.notebook} to {output_path}")


if __name__ == "__main__":
    main()