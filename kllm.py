#!/usr/bin/env python3
"""Run LLMs from Kaggle Models, either on this Mac (MLX) or on Kaggle's free GPUs.

  kllm.py local  [HANDLE] "prompt"   download via kagglehub, run on Apple Silicon with mlx-lm
  kllm.py remote [HANDLE] "prompt"   push a GPU script to Kaggle, wait, print the result
  (HANDLE is optional; defaults are Qwen 3 1.7B locally and Qwen 3 4B remotely)
  kllm.py status / kllm.py fetch   check or re-download the last remote run

  kllm.py worker        JSON-lines local chat server on stdin/stdout (used by the desktop app)
  kllm.py chat-remote   one remote chat turn, JSON request on stdin, JSON events on stdout

HANDLE is a Kaggle model handle: owner/model/framework/variation[/version],
e.g. google/gemma-3/transformers/gemma-3-1b-it
"""
import argparse, json, os, subprocess, sys, tempfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KAGGLE = str(ROOT / ".venv" / "bin" / "kaggle")
STATE = ROOT / ".last_remote.json"
OUT = ROOT / "outputs"
# Used when only a prompt is given. Both are ungated (no license to accept).
DEFAULTS = {"local": "qwen-lm/qwen-3/transformers/1.7b",   # light enough for a 16GB Mac
            "remote": "qwen-lm/qwen-3/transformers/4b"}    # ~8GB fp16, fits a Kaggle T4
# Sampling settings recommended for Qwen 3; greedy decoding makes it loop.
TEMP, TOP_P = 0.6, 0.95


class KllmError(Exception):
    pass


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


# ---------------------------------------------------------------- local (MLX)

_loaded = {}  # handle -> (model, tokenizer); only the most recent one is kept


def load_local(handle):
    from mlx_lm import load
    if handle not in _loaded:
        _loaded.clear()
        _loaded[handle] = load(download(handle))
    return _loaded[handle]


def stream_local(handle, messages, max_tokens):
    """Yield text chunks for a chat completion on this Mac."""
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    model, tok = load_local(handle)
    prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    for r in stream_generate(model, tok, prompt, max_tokens=max_tokens,
                             sampler=make_sampler(temp=TEMP, top_p=TOP_P)):
        yield r.text


def local(args):
    for chunk in stream_local(args.handle, [{"role": "user", "content": args.prompt}], args.max_tokens):
        print(chunk, end="", flush=True)
    print()


def worker():
    """Serve local chat requests: one JSON object per stdin line
    {"id", "handle", "messages", "max_tokens"}; reply with JSON lines
    {"id", "type": status|token|done|error, ...}. The model stays loaded between requests."""
    out = sys.stdout
    sys.stdout = sys.stderr  # libraries print progress; keep stdout for protocol only

    def emit(**ev):
        out.write(json.dumps(ev) + "\n"); out.flush()

    emit(type="ready")
    for line in sys.stdin:
        if not line.strip():
            continue
        req = json.loads(line)
        rid, handle = req.get("id"), req.get("handle") or DEFAULTS["local"]
        try:
            if handle not in _loaded:
                emit(id=rid, type="status", text=f"Loading {handle.split('/')[-1]} (first time downloads it)…")
            started = False
            for chunk in stream_local(handle, req["messages"], int(req.get("max_tokens", 2048))):
                if not started:
                    emit(id=rid, type="status", text=""); started = True
                emit(id=rid, type="token", text=chunk)
            emit(id=rid, type="done")
        except Exception as e:  # report and keep serving
            emit(id=rid, type="error", text=f"{type(e).__name__}: {e}")


# ------------------------------------------------------------- remote (Kaggle)

REMOTE_SCRIPT = '''
import json, torch, kagglehub
from transformers import AutoTokenizer, AutoModelForCausalLM

HANDLE, MESSAGES, MAX_TOKENS = {handle!r}, {messages!r}, {max_tokens}
path = kagglehub.model_download(HANDLE)  # resolves to the attached /kaggle/input copy
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16, device_map="auto")
inputs = tok.apply_chat_template(MESSAGES, add_generation_prompt=True, return_tensors="pt",
                                 return_dict=True).to(model.device)
out = model.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=True,
                     temperature={temp}, top_p={top_p})
text = tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
print(text)
json.dump({{"handle": HANDLE, "messages": MESSAGES, "response": text}},
          open("/kaggle/working/result.json", "w"), indent=2)
'''


def kaggle(*a):
    r = subprocess.run([KAGGLE, *a], text=True, capture_output=True)
    if r.returncode:
        raise KllmError((r.stderr or r.stdout).strip() or f"kaggle {a[0]} {a[1]} failed")
    return r.stdout


def username():
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    u = api.config_values.get("username")
    if not u:
        raise KllmError("Could not determine your Kaggle username; set KAGGLE_USERNAME.")
    return u


def push(handle, messages, max_tokens, slug):
    kid = f"{username()}/{slug}"
    with tempfile.TemporaryDirectory() as d:
        Path(d, "run.py").write_text(REMOTE_SCRIPT.format(
            handle=handle, messages=messages, max_tokens=max_tokens, temp=TEMP, top_p=TOP_P))
        # Model sources need an explicit version; default to 1 if none given.
        src = handle if handle.count("/") == 4 else handle + "/1"
        Path(d, "kernel-metadata.json").write_text(json.dumps({
            "id": kid, "title": slug, "code_file": "run.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True,
            "enable_internet": True, "model_sources": [src],
            "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
        }, indent=2))
        kaggle("kernels", "push", "-p", d)
    STATE.write_text(json.dumps({"id": kid}))
    return kid


def wait(kid, on_status=lambda s: None):
    """Poll until the run finishes; returns the status word (complete/error/cancel...)."""
    while True:
        s = kaggle("kernels", "status", kid).lower()
        word = s.rsplit("kernelworkerstatus.", 1)[-1].strip(' "\n') if "kernelworkerstatus." in s else s.strip()
        on_status(word)
        if any(w in word for w in ("complete", "error", "cancel")):
            return word
        time.sleep(10)


def fetch(kid):
    """Download outputs; return (response or None, path to the log)."""
    dest = OUT / kid.split("/")[1]
    dest.mkdir(parents=True, exist_ok=True)
    for f in dest.glob("*"):
        f.unlink()  # never return a stale result from an earlier run
    try:
        kaggle("kernels", "output", kid, "-p", str(dest), "-o")
    except KllmError:
        pass
    res = dest / "result.json"
    return (json.loads(res.read_text())["response"] if res.exists() else None), dest / f"{kid.split('/')[1]}.log"


def log_tail(log, n=12):
    """Last lines of a Kaggle run log (stored as JSON list of stream chunks)."""
    try:
        text = "".join(x.get("data", "") for x in json.loads(log.read_text()))
    except Exception:
        return ""
    lines = [l for l in text.splitlines() if "SyntaxWarning" not in l and "NbConvertApp" not in l
             and not l.strip().startswith(("cells[i]", "text = re.sub"))]
    return "\n".join(lines[-n:])


def remote(args):
    kid = push(args.handle, [{"role": "user", "content": args.prompt}], args.max_tokens, args.slug or "kllm-run")
    print(f"Pushed https://www.kaggle.com/code/{kid}", file=sys.stderr)
    if args.no_wait:
        return
    word = wait(kid, lambda s: print(f"  {s}", file=sys.stderr))
    text, log = fetch(kid)
    if text is None:
        raise KllmError(f"Remote run ended with status {word}.\n{log_tail(log)}")
    print(text)


def chat_remote():
    """One remote chat turn for the app. Reads {"handle","messages","max_tokens","slug"} on stdin,
    writes JSON lines {"type": status|done|error, "text", ...} on stdout."""
    out = sys.stdout
    sys.stdout = sys.stderr

    def emit(**ev):
        out.write(json.dumps(ev) + "\n"); out.flush()

    req = json.loads(sys.stdin.read())
    t0 = time.time()
    try:
        emit(type="status", text="Uploading notebook to Kaggle…")
        kid = push(req.get("handle") or DEFAULTS["remote"], req["messages"],
                   int(req.get("max_tokens", 2048)), req.get("slug") or "kllm-app")
        emit(type="status", text="Queued on Kaggle…", url=f"https://www.kaggle.com/code/{kid}")
        word = wait(kid, lambda s: emit(type="status", text=f"Kaggle: {s} · {int(time.time() - t0)}s",
                                         url=f"https://www.kaggle.com/code/{kid}"))
        emit(type="status", text="Downloading result…")
        text, log = fetch(kid)
        if text is None:
            raise KllmError(f"Kaggle run ended with status '{word}'.\n{log_tail(log)}")
        emit(type="done", text=text, seconds=int(time.time() - t0))
    except Exception as e:
        emit(type="error", text=str(e))


def last_id():
    if not STATE.exists():
        raise KllmError("No remote run yet.")
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
    for name in ("status", "fetch", "worker", "chat-remote"):
        sub.add_parser(name)
    a = p.parse_args()
    if a.cmd in DEFAULTS:
        if len(a.args) > 2:
            p.error('too many arguments; put the prompt in quotes: kllm.py remote "your prompt"')
        a.handle, a.prompt = a.args if len(a.args) == 2 else (DEFAULTS[a.cmd], a.args[0])
    try:
        if a.cmd == "local": local(a)
        elif a.cmd == "remote": remote(a)
        elif a.cmd == "status": print(kaggle("kernels", "status", last_id()).strip())
        elif a.cmd == "fetch":
            text, log = fetch(last_id())
            print(text if text is not None else log_tail(log))
        elif a.cmd == "worker": worker()
        else: chat_remote()
    except KllmError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
