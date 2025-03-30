import json
import argparse
import os
from pathlib import Path


def convert_notebook_to_script(notebook_path, output_path=None, include_markdown=True, include_outputs=True):
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
            
            if include_outputs and 'outputs' in cell and cell['outputs']:
                script_content.append("")
                script_content.append("# " + "-" * 40)
                script_content.append("# OUTPUT:")
                script_content.append("# " + "-" * 40)
                
                for output in cell['outputs']:
                    if 'text' in output:
                        for line in output['text']:
                            script_content.append(f"# {line.rstrip()}")
                    elif 'data' in output:
                        if 'text/plain' in output['data']:
                            text_output = output['data']['text/plain']
                            if isinstance(text_output, list):
                                for line in text_output:
                                    script_content.append(f"# {line.rstrip()}")
                            else:
                                script_content.append(f"# {text_output}")

                        for output_type in output['data']:
                            if output_type != 'text/plain':
                                script_content.append(f"# [Output type: {output_type}]")
                    elif 'traceback' in output:
                        script_content.append("# [Error occurred]")
                        for line in output['traceback']:
                            clean_line = line.replace('\u001b[0;31m', '').replace('\u001b[0;32m', '')
                            clean_line = clean_line.replace('\u001b[0m', '').replace('\u001b[1;36m', '')
                            script_content.append(f"# {clean_line.rstrip()}")
                            
                script_content.append("# " + "-" * 40)
        
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
    parser.add_argument('--no-outputs', action='store_true',
                        help='Skip code cell outputs in the output')
    
    args = parser.parse_args()
    
    output_path = convert_notebook_to_script(
        args.notebook, 
        args.output, 
        not args.no_markdown,
        not args.no_outputs
    )
    
    print(f"Converted {args.notebook} to {output_path}")


if __name__ == "__main__":
    main()