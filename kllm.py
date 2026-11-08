#!/usr/bin/env python3
"""Run LLMs from Kaggle Models, either on this Mac (MLX) or on Kaggle's free GPUs.

  kllm.py local  [HANDLE] "prompt"   download via kagglehub, run on Apple Silicon with mlx-lm
  kllm.py remote [HANDLE] "prompt"   push a GPU script to Kaggle, wait, print the result
  (HANDLE is optional; defaults are Qwen 3 1.7B locally and Qwen 3 4B remotely)
  kllm.py status / kllm.py fetch   check or re-download the last remote run

HANDLE is a Kaggle model handle: owner/model/framework/variation[/version],
e.g. google/gemma-3/transformers/gemma-3-1b-it
"""
import argparse, json, os, subprocess, sys, tempfile, textwrap, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KAGGLE = str(ROOT / ".venv" / "bin" / "kaggle")
STATE = ROOT / ".last_remote.json"
OUT = ROOT / "outputs"
# Used when only a prompt is given. Both are ungated (no license to accept).
DEFAULTS = {"local": "qwen-lm/qwen-3/transformers/1.7b",   # light enough for a 16GB Mac
            "remote": "qwen-lm/qwen-3/transformers/4b"}    # ~8GB fp16, fits a Kaggle T4


def download(handle):
    """kagglehub.model_download, with a fix for a kagglesdk bug: it treats any JSON
    response containing "code" >= 400 as an API error -- including downloaded model
    files such as a tokenizer vocab.json where "code" and "message" are just tokens.
    Only apply that check to responses from the Kaggle API itself, not file storage."""
    import kagglehub
    from kagglesdk.kaggle_http_client import KaggleHttpClient

    orig = KaggleHttpClient._prepare_response
    if not getattr(orig, "_kllm_patched", False):
        def patched(self, response_type, http_response):
            if "kaggle.com" not in (http_response.url or "").split("/")[2]:
                http_response.raise_for_status()
                return response_type.prepare_from(http_response) if response_type else None
            return orig(self, response_type, http_response)
        patched._kllm_patched = True
        KaggleHttpClient._prepare_response = patched
    return kagglehub.model_download(handle)


def local(args):
    from mlx_lm import load, generate

    path = download(args.handle)
    print(f"[model at {path}]", file=sys.stderr)
    model, tok = load(path)
    prompt = args.prompt
    if tok.chat_template:
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
    generate(model, tok, prompt=prompt, max_tokens=args.max_tokens, verbose=True)


REMOTE_SCRIPT = '''
import json, torch, kagglehub
from transformers import AutoTokenizer, AutoModelForCausalLM

HANDLE, PROMPT, MAX_TOKENS = {handle!r}, {prompt!r}, {max_tokens}
path = kagglehub.model_download(HANDLE)  # resolves to the attached /kaggle/input copy
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16, device_map="auto")
msgs = [{{"role": "user", "content": PROMPT}}]
inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                 return_dict=True).to(model.device)
out = model.generate(**inputs, max_new_tokens=MAX_TOKENS)
text = tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
print(text)
json.dump({{"handle": HANDLE, "prompt": PROMPT, "response": text}}, open("/kaggle/working/result.json", "w"), indent=2)
'''


def kaggle(*a, check=True, capture=False):
    r = subprocess.run([KAGGLE, *a], text=True, capture_output=capture)
    if check and r.returncode:
        sys.exit(r.stderr if capture else r.returncode)
    return r.stdout if capture else r


def username():
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    u = api.config_values.get("username")
    if not u:
        sys.exit("Could not determine your Kaggle username; set KAGGLE_USERNAME.")
    return u


def wait_and_fetch(kid):
    print(f"Waiting for https://www.kaggle.com/code/{kid} ...", file=sys.stderr)
    while True:
        s = kaggle("kernels", "status", kid, capture=True).lower()
        if "complete" in s:
            break
        if "error" in s or "cancel" in s:
            print(s, file=sys.stderr)
            fetch(kid)
            sys.exit("Remote run failed; see the log in outputs/.")
        time.sleep(20)
    fetch(kid)


def fetch(kid):
    dest = OUT / kid.split("/")[1]
    dest.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", kid, "-p", str(dest), "-o", check=False)
    res = dest / "result.json"
    if res.exists():
        print(json.loads(res.read_text())["response"])
    print(f"[outputs in {dest}]", file=sys.stderr)


def remote(args):
    user = username()
    slug = args.slug or "kllm-run"
    kid = f"{user}/{slug}"
    with tempfile.TemporaryDirectory() as d:
        Path(d, "run.py").write_text(REMOTE_SCRIPT.format(
            handle=args.handle, prompt=args.prompt, max_tokens=args.max_tokens))
        # Model sources need an explicit version; default to 1 if none given.
        src = args.handle if args.handle.count("/") == 4 else args.handle + "/1"
        Path(d, "kernel-metadata.json").write_text(json.dumps({
            "id": kid, "title": slug, "code_file": "run.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True,
            "enable_internet": True, "model_sources": [src],
            "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
        }, indent=2))
        kaggle("kernels", "push", "-p", d)
    STATE.write_text(json.dumps({"id": kid}))
    if not args.no_wait:
        wait_and_fetch(kid)


def last_id():
    if not STATE.exists():
        sys.exit("No remote run yet.")
    return json.loads(STATE.read_text())["id"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("local", "remote"):
        s = sub.add_parser(name, usage=f"%(prog)s [HANDLE] PROMPT  (default model: {DEFAULTS[name]})")
        s.add_argument("args", nargs="+", metavar="[HANDLE] PROMPT")
        s.add_argument("--max-tokens", type=int, default=512)
        if name == "remote":
            s.add_argument("--slug", help="Kaggle notebook name (default kllm-run)")
            s.add_argument("--no-wait", action="store_true")
    sub.add_parser("status"); sub.add_parser("fetch")
    a = p.parse_args()
    if a.cmd in DEFAULTS:
        if len(a.args) > 2:
            p.error('too many arguments; put the prompt in quotes: kllm.py remote "your prompt"')
        a.handle, a.prompt = a.args if len(a.args) == 2 else (DEFAULTS[a.cmd], a.args[0])
    if a.cmd == "local": local(a)
    elif a.cmd == "remote": remote(a)
    elif a.cmd == "status": kaggle("kernels", "status", last_id())
    else: fetch(last_id())


if __name__ == "__main__":
    main()
