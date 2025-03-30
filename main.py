from jupyter_to_python import convert_notebook_to_script

# Convert with default settings
convert_notebook_to_script('example.ipynb')

# Customize conversion
# convert_notebook_to_script('example.ipynb', 
#                           output_path='custom_name.py',
#                           include_markdown=False)