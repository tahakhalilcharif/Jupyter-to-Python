from jupyter_to_python import convert_notebook_to_script

# Convert with default settings
convert_notebook_to_script('cal_housing.ipynb')

# Customize conversion
# convert_notebook_to_script('my_notebook.ipynb', 
#                           output_path='custom_name.py',
#                           include_markdown=False)