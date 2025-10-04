# data_analysis
UBC Solar's data analysis environment


## Requirements
1. Git (`git --version`) 
2. Git LFS (`git lfs --version`).
   1. Install with `brew install git-lfs` or similar.
3. Python 3.11 (`python3 --version`) 
4. Poetry (`poetry --version`)

## Install Dependencies
1. `poetry install --no-root`

## Troubleshooting

If your Jupyter kernel fails to resolve packages in the Poetry environment such as `data_tools` and `physics`, you may need to run
```bash
python -m ipykernel install --user --name=stg-data-analysis --display-name "stg-data-anaylsis"
```
and then select the "stg-data-anaylsis" kernel instead of the base "ipykernel". 
