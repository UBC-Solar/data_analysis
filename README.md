# Data Analysis
UBC Solar's data analysis environment!

To begin, make a new branch with a relevant name and make a folder with the same name.
```bash
git checkout -b my_new_data_analysis_project
mkdir my_new_data_analysis_project
```

Please ensure that you follow the [data analysis guidelines](https://wiki.ubcsolar.com/en/subteams/simulation/data-analysis-guidelines).

## Requirements
1. Git (`git --version`) 
2. Git LFS (`git lfs --version`).
   1. Install with `brew install git-lfs` or similar.
3. Python >3.11 (`python3 --version`) 
4. Poetry (`poetry --version`)

## Install Dependencies
1. `poetry install --no-root`

If your Jupyter environment fails to resolve the `ubc-solar-data-tools` package, you may need to run `register_kernel.sh` to register the Poetry environment as a Jupyter kernel, and switch to using it.

## Conventions

1. Any data analysis tools that don't belong in `ubc-solar-data-tools` should go into the `tools/` folder.
2. Local data, such as a `.csv` file that is being analyzed, should be put into a `data/` folder inside of your analysis folder. For example, `my_awesome_data.csv` should be located in `your_project/data/`.