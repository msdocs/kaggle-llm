# kaggle-llm

Run open LLMs from [Kaggle Models](https://www.kaggle.com/models) in one of two ways:

- **local**: download the model with `kagglehub` and run it on Apple Silicon with [MLX](https://github.com/ml-explore/mlx-lm)
- **remote**: push a private GPU notebook to Kaggle (free T4/P100, about 30 hrs/week), wait for it to finish, and print the answer

It's also the backend for [Parley](https://mahmoud.sh/parley/), a desktop chat app for macOS.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Create an API token at kaggle.com → Settings → API → **Generate New Token**, then save it:

```bash
mkdir -p ~/.kaggle && echo "YOUR_TOKEN" > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token
```

Local mode needs an Apple Silicon Mac. Remote mode works from any machine.

## Usage

```bash
.venv/bin/python kllm.py local  "your prompt"     # Qwen 3 1.7B on your Mac
.venv/bin/python kllm.py remote "your prompt"     # Qwen 3 4B on a Kaggle GPU
.venv/bin/python kllm.py remote google/gemma-3/transformers/gemma-3-4b-it "your prompt"
.venv/bin/python kllm.py status                   # status of the last remote run
.venv/bin/python kllm.py fetch                    # re-download its outputs
```

Options: `--max-tokens N` (default 512); for remote only, `--slug NAME` (notebook name, default `kllm-run`) and `--no-wait`.

A model handle has the form `owner/model/framework/variation[/version]`. Use the **Transformers** variant from the model's Kaggle page. Gated models such as Gemma and Llama require you to accept their license on Kaggle first.

## Where things are stored

- **Local models:** `~/.cache/kagglehub/models/` (set `KAGGLEHUB_CACHE` to use a different folder)
- **Remote models:** attached on Kaggle's servers at `/kaggle/input`; nothing is downloaded to your machine
- **Remote results:** `outputs/<slug>/result.json` and the run log

## Rough size limits

- Local, on a 16 GB Mac: up to about 4B parameters in fp16, or 8B if quantized
- Remote, on a Kaggle T4 (16 GB): up to about 7–8B in fp16

## Notes

`kllm.py` patches a bug in `kagglesdk`: it treats any downloaded JSON file containing `"code" >= 400` and `"message"` as an API error. Some tokenizer `vocab.json` files contain both words as tokens, so downloads of those models failed.
