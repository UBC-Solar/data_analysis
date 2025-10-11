# Results

Use this folder to store the results of your analysis,
like PDF files, images, tables or other data.

---

Jupyter notebooks alone can be complete enough as results,
but consider exporting them to PDF files to make the result more accessible.

To export a Jupyter notebook to pdf, the quickest way is to use a web converter.
https://onlineconvertfree.com/convert/ipynb/ works pretty well.

A command-line tool can be helpful if you want more control over the output or need to convert many files.
To convert to PDF on the command-line,

1. Install `nbconvert`

`uv add nbconvert`

You'll also need to install [Pandoc](https://minrk-nbconvert.readthedocs.io/en/stable/install.html#installing-pandoc) to convert to PDF.

2. Export the file

`\your_project> nbconvert .\example_notebook.ipynb --to pdf`