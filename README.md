# king-county

Project 2 for Data Science II.

## Files

- `kc_house_data.csv`: the dataset.
- `model.py`: main training script.
- `plot.py`: loss plots for each hyperparameter configuration.
- `plot_params.py`: box plots for each hyperparameter value.

## Requirements

- Python 3.14 (older versions may work)
- CUDA-capable GPU

## Run

I recommend using [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
cd king-county-homes
uv venv
uv pip install -r requirements.txt
uv run model.py
```

Otherwise, just:

```bash
cd king-county-homes
pip install -r requirements.txt
python3 model.py
```
