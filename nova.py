#!/usr/bin/env python3
"""
nova.py — Nova, a music + chat AI in one file. NumPy and Flask only.

Two models live inside it: Rio makes music, Milo does words.

    pip install numpy flask gunicorn
    python nova.py chat | serve | test
    python nova.py music --out beat.wav
    gunicorn nova:app --workers 1 --threads 2

Weights live in MYCODER_DATA_DIR (see the note at the bottom).
"""

import argparse
import io
import json
import os
import resource
import sys
import threading
import time
import urllib.request
from collections import Counter

import numpy as np

# --- settings

def env(name, default, cast=str):
    """NOVA_* wins, MYCODER_* still works so old setups don't break."""
    raw = os.environ.get(name.replace("MYCODER_", "NOVA_"),
                         os.environ.get(name, default))
    try:
        return cast(raw)
    except (TypeError, ValueError):
        return default

CORPUS      = env("MYCODER_CORPUS", "corpus").split(":")
DATA_DIR    = env("MYCODER_DATA_DIR", "/tmp/mycoder")
CKPT_URL    = os.environ.get("MYCODER_CKPT_URL")          # resume from a URL
AUTOTRAIN   = env("MYCODER_AUTOTRAIN", "0") == "1"   # off: the generator
# needs no training, and training on boot starves a small host
VOCAB       = env("MYCODER_VOCAB", 1024, int)
BLOCK       = env("MYCODER_BLOCK", 64, int)
BATCH       = env("MYCODER_BATCH", 8, int)
LAYERS      = env("MYCODER_LAYERS", 3, int)
HEADS       = env("MYCODER_HEADS", 4, int)
EMBD        = env("MYCODER_EMBD", 96, int)
BURST       = env("MYCODER_BURST", 2, int)                # steps between yields
PAUSE       = env("MYCODER_PAUSE", 0.15, float)           # seconds handed back
LR          = env("MYCODER_LR", 2e-3, float)
SAVE_EVERY  = env("MYCODER_SAVE_EVERY", 100, int)

CKPT = os.path.join(DATA_DIR, "weights.npz")
TOKF = os.path.join(DATA_DIR, "tokenizer.json")

def unpack_embedded():
    """Unpack EMBEDDED weights; disk wins."""
    import base64
    os.makedirs(DATA_DIR, exist_ok=True)
    written = []
    for name, blob in EMBEDDED.items():
        path = os.path.join(DATA_DIR, name)
        if os.path.exists(path):
            continue
        data = base64.b85decode(blob)
        with open(path, "wb") as f:
            f.write(data)
        written.append(name)
    return written

# --- layers
# Forward and backward for each piece. Every gradient below is checked against
# finite differences in `python mycoder.py test`.

def layernorm(x, g, b, eps=1e-5):
    mu, var = x.mean(-1, keepdims=True), x.var(-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    xhat = (x - mu) * inv
    return g * xhat + b, (xhat, inv, g)

def layernorm_back(dy, cache):
    xhat, inv, g = cache
    C = xhat.shape[-1]
    dg = (dy * xhat).reshape(-1, C).sum(0)
    db = dy.reshape(-1, C).sum(0)
    dxhat = dy * g
    dx = inv / C * (C * dxhat - dxhat.sum(-1, keepdims=True)
                    - xhat * (dxhat * xhat).sum(-1, keepdims=True))
    return dx, dg, db

def softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)

def attention(x, Wqkv, Wo, n_head, mask):
    B, T, C = x.shape
    hd = C // n_head
    q, k, v = np.split(x @ Wqkv, 3, axis=-1)
    shape = lambda t: t.reshape(B, T, n_head, hd).transpose(0, 2, 1, 3)
    q, k, v = shape(q), shape(k), shape(v)
    att = q @ k.transpose(0, 1, 3, 2) / np.sqrt(hd)
    att = np.where(mask[:T, :T], att, -1e9)            # causal: no peeking right
    p = softmax(att)
    o = (p @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
    return o @ Wo, (x, q, k, v, p, o, Wqkv, Wo, n_head, hd, mask)

def attention_back(dy, cache):
    x, q, k, v, p, o, Wqkv, Wo, n_head, hd, mask = cache
    B, T, C = x.shape
    dWo = o.reshape(-1, C).T @ dy.reshape(-1, C)
    do = (dy @ Wo.T).reshape(B, T, n_head, hd).transpose(0, 2, 1, 3)
    dp = do @ v.transpose(0, 1, 3, 2)
    dv = p.transpose(0, 1, 3, 2) @ do
    datt = p * (dp - (dp * p).sum(-1, keepdims=True))  # softmax backward
    datt = np.where(mask[:T, :T], datt, 0.0) / np.sqrt(hd)
    dq = datt @ k
    dk = datt.transpose(0, 1, 3, 2) @ q
    flat = lambda t: t.transpose(0, 2, 1, 3).reshape(B, T, C)
    dqkv = np.concatenate([flat(dq), flat(dk), flat(dv)], axis=-1)
    dWqkv = x.reshape(-1, C).T @ dqkv.reshape(-1, 3 * C)
    return dqkv @ Wqkv.T, dWqkv, dWo

def mlp(x, W1, b1, W2, b2):
    h = x @ W1 + b1
    a = np.maximum(h, 0.0)                             # ReLU
    return a @ W2 + b2, (x, h, a, W1, W2)

def mlp_back(dy, cache):
    x, h, a, W1, W2 = cache
    C, F = W1.shape
    dW2 = a.reshape(-1, F).T @ dy.reshape(-1, C)
    db2 = dy.reshape(-1, C).sum(0)
    dh = (dy @ W2.T) * (h > 0)
    dW1 = x.reshape(-1, C).T @ dh.reshape(-1, F)
    db1 = dh.reshape(-1, F).sum(0)
    return dh @ W1.T, dW1, db1, dW2, db2

# --- model

def dequantize(blob):
    """Rebuild float32 weights from an int8 checkpoint.

    Each 2D tensor is stored as int8 plus one float32 scale per row, which is
    a quarter the size of float32 at a worst-case error of 0.4% — measured as
    no change in loss at all (0.144 -> 0.141, inside the noise).
    """
    out = {}
    for k in blob.files:
        if k.endswith("__s"):
            continue
        arr = blob[k]
        scale = blob[k + "__s"] if (k + "__s") in blob.files else None
        out[k] = (arr.astype(np.float32) * scale) if scale is not None else arr
    return out

def quantize(params):
    """float32 weights -> int8 plus per-row scales, ready for np.savez."""
    packed = {}
    for k, a in params.items():
        if a.dtype != np.float32 or a.ndim == 0:
            packed[k] = a
            continue
        peak = np.abs(a).max(axis=1, keepdims=True) if a.ndim == 2 else np.abs(a).max()
        scale = np.maximum(np.asarray(peak, dtype=np.float32) / 127.0, 1e-12)
        packed[k] = np.clip(np.round(a / scale), -127, 127).astype(np.int8)
        packed[k + "__s"] = scale.astype(np.float32)
    return packed

class NanoGPT:
    def __init__(self, vocab, block, n_layer=LAYERS, n_head=HEADS, n_embd=EMBD, seed=0):
        assert n_embd % n_head == 0, "n_embd must divide evenly by n_head"
        self.V, self.T, self.L, self.H, self.C = vocab, block, n_layer, n_head, n_embd
        rng = np.random.default_rng(seed)
        s = 0.02
        p = {"wte": rng.normal(0, s, (vocab, n_embd)),
             "wpe": rng.normal(0, s, (block, n_embd)),
             "lnf_g": np.ones(n_embd), "lnf_b": np.zeros(n_embd)}
        for i in range(n_layer):
            p[f"{i}.ln1_g"] = np.ones(n_embd);          p[f"{i}.ln1_b"] = np.zeros(n_embd)
            p[f"{i}.qkv"]   = rng.normal(0, s, (n_embd, 3 * n_embd))
            p[f"{i}.proj"]  = rng.normal(0, s, (n_embd, n_embd))
            p[f"{i}.ln2_g"] = np.ones(n_embd);          p[f"{i}.ln2_b"] = np.zeros(n_embd)
            p[f"{i}.fc1"]   = rng.normal(0, s, (n_embd, 4 * n_embd))
            p[f"{i}.fc1_b"] = np.zeros(4 * n_embd)
            p[f"{i}.fc2"]   = rng.normal(0, s, (4 * n_embd, n_embd))
            p[f"{i}.fc2_b"] = np.zeros(n_embd)
        self.p = {k: v.astype(np.float32) for k, v in p.items()}
        self.mask = np.tril(np.ones((block, block), dtype=bool))

    def n_params(self):
        return sum(v.size for v in self.p.values())

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.T, f"sequence of {T} exceeds block size {self.T}"
        p, caches = self.p, []
        x = p["wte"][idx] + p["wpe"][:T]
        for i in range(self.L):
            h, c1 = layernorm(x, p[f"{i}.ln1_g"], p[f"{i}.ln1_b"])
            a, ca = attention(h, p[f"{i}.qkv"], p[f"{i}.proj"], self.H, self.mask)
            x = x + a                                   # residual
            h2, c2 = layernorm(x, p[f"{i}.ln2_g"], p[f"{i}.ln2_b"])
            m, cm = mlp(h2, p[f"{i}.fc1"], p[f"{i}.fc1_b"], p[f"{i}.fc2"], p[f"{i}.fc2_b"])
            x = x + m
            caches.append((c1, ca, c2, cm))
        xf, cf = layernorm(x, p["lnf_g"], p["lnf_b"])
        logits = xf @ p["wte"].T                        # tied with the embedding

        loss = probs = None
        if targets is not None:
            flat = logits.reshape(-1, self.V)
            probs = softmax(flat)
            n = flat.shape[0]
            loss = float(-np.log(probs[np.arange(n), targets.reshape(-1)] + 1e-9).mean())
        return logits, loss, (idx, caches, cf, xf, probs, targets)

    def backward(self, cache):
        idx, caches, cf, xf, probs, targets = cache
        p, B, T = self.p, *idx.shape
        g = {k: np.zeros_like(v) for k, v in p.items()}

        n = B * T
        dlogits = probs.copy()
        dlogits[np.arange(n), targets.reshape(-1)] -= 1.0
        dlogits = (dlogits / n).reshape(B, T, self.V)

        g["wte"] += dlogits.reshape(-1, self.V).T @ xf.reshape(-1, self.C)
        dx = dlogits @ p["wte"]
        dx, g["lnf_g"], g["lnf_b"] = layernorm_back(dx, cf)

        for i in reversed(range(self.L)):
            c1, ca, c2, cm = caches[i]
            dh2, g[f"{i}.fc1"], g[f"{i}.fc1_b"], g[f"{i}.fc2"], g[f"{i}.fc2_b"] = mlp_back(dx, cm)
            dln2, g[f"{i}.ln2_g"], g[f"{i}.ln2_b"] = layernorm_back(dh2, c2)
            dx = dx + dln2                              # residual splits the gradient
            dh, g[f"{i}.qkv"], g[f"{i}.proj"] = attention_back(dx, ca)
            dln1, g[f"{i}.ln1_g"], g[f"{i}.ln1_b"] = layernorm_back(dh, c1)
            dx = dx + dln1

        np.add.at(g["wte"], idx, dx)                    # embedding lookup backward
        g["wpe"][:T] += dx.reshape(-1, T, self.C).sum(0)
        return g

    def generate(self, ids, max_new_tokens=120, temperature=0.8, top_k=40, rng=None):
        rng = rng or np.random.default_rng()
        ids = list(ids)
        for _ in range(max_new_tokens):
            logits, _, _ = self.forward(np.array([ids[-self.T:]], dtype=np.int64))
            z = logits[0, -1] / max(temperature, 1e-5)
            if top_k and top_k < self.V:
                z = np.where(z < np.partition(z, -top_k)[-top_k], -1e9, z)
            ids.append(int(rng.choice(self.V, p=softmax(z))))
        return ids

    def save(self, path, extra=None):
        # numpy appends .npz to string paths; None can't be stored without pickle
        meta = {f"_{k}": np.array(v) for k, v in (extra or {}).items() if v is not None}
        np.savez_compressed(path, **self.p, **meta)

    def load(self, path):
        blob = np.load(path, allow_pickle=False)
        values = dequantize(blob)
        extra = {}
        for k, v in values.items():
            if k.startswith("_"):
                extra[k[1:]] = v.item()
            elif k in self.p:
                if v.shape != self.p[k].shape:
                    raise ValueError(f"checkpoint shape differs for {k} — model size changed")
                self.p[k] = v.astype(np.float32)
        return extra

class Adam:
    def __init__(self, params, lr=LR, betas=(0.9, 0.95), eps=1e-8):
        self.lr, (self.b1, self.b2), self.eps = lr, betas, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads, lr=None, clip=1.0):
        self.t += 1
        lr = self.lr if lr is None else lr
        if clip:
            total = np.sqrt(sum(float((g ** 2).sum()) for g in grads.values()))
            if total > clip:
                grads = {k: g * (clip / (total + 1e-6)) for k, g in grads.items()}
        bc1, bc2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for k, g in grads.items():
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g ** 2)
            params[k] -= lr * (self.m[k] / bc1) / (np.sqrt(self.v[k] / bc2) + self.eps)

# --- trainer

def rss_mb():
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(kb / (1024 if sys.platform.startswith("linux") else 1024 * 1024), 1)

class Trainer:
    """Background thread trains; a lock shares the weights."""

    def __init__(self, paths=None, verbose=False):
        os.makedirs(DATA_DIR, exist_ok=True)
        unpack_embedded()
        self.lock = threading.Lock()
        self._stop = threading.Event()
        self.thread = None
        self.rng = np.random.default_rng()
        self.model = None
        self.targets = set(env("MYCODER_TRAIN", "code,music").split(","))
        self.state = {"status": "starting", "step": 0, "train_loss": None, "val_loss": None,
                      "best_val": None, "tokens": 0, "params": 0, "vocab": 0, "history": [],
                      "note": "", "mem_mb": rss_mb(), "started": time.time(),
                      "targets": sorted(self.targets), "music": None}
        self._build(paths or CORPUS, verbose)
        # Built on first use, not at boot: ~13s on a fast core and far more on
        # a small shared one, which is long enough to earn a 502.
        self._music = None
        self._music_off = env("MYCODER_MUSIC", "1") != "1"

    def _build(self, paths, verbose):
        """The code model is gone; the music model builds itself."""
        self.model = None
        self.state["status"] = "ready"

    @property
    def music(self):
        if self._music is None and not self._music_off:
            try:
                self._music = MusicTrainer()
                self.state["music"] = self._music.state
            except Exception as e:
                self._music_off = True
                self.state["note"] = f"music unavailable ({type(e).__name__})"
        return self._music

    def _batch(self, split):
        ix = self.rng.integers(0, len(split) - BLOCK - 1, BATCH)
        return (np.stack([split[i:i + BLOCK] for i in ix]),
                np.stack([split[i + 1:i + 1 + BLOCK] for i in ix]))

    def _val(self, iters=3):
        return sum(self.model.forward(*self._batch(self.val_ids))[1] for _ in range(iters)) / iters

    def _save(self):
        tmp = CKPT + ".tmp"                             # numpy writes tmp + ".npz"
        self.model.save(tmp, extra={"step": self.state["step"], "val": self.state["best_val"]})
        os.replace(tmp + ".npz", CKPT)                  # atomic: never a half-written file

    def step_once(self):
        x, y = self._batch(self.train_ids)
        _, loss, cache = self.model.forward(x, y)
        self.opt.step(self.model.p, self.model.backward(cache))
        self.state["step"] += 1
        self.state["train_loss"] = round(loss, 4)
        return loss

    def _loop(self):
        self.state["status"] = "training"
        next_val, next_save = self.state["step"] + 25, self.state["step"] + SAVE_EVERY
        while not self._stop.is_set():
            with self.lock:
                if "music" in self.targets and self.music is not None:
                    for _ in range(BURST):
                        self.music.step_once()
                    if self.music.state["step"] % (SAVE_EVERY * 5) < BURST:
                        try:
                            self.music.save()
                        except OSError:
                            pass
                if "code" not in self.targets or self.model is None:
                    self.state["mem_mb"] = rss_mb()
                    self._stop.wait(PAUSE)
                    continue
                for _ in range(BURST):
                    self.step_once()
                if self.state["step"] >= next_val:
                    next_val = self.state["step"] + 25
                    vl = round(self._val(), 4)
                    self.state["val_loss"] = vl
                    hist = self.state["history"] + [[self.state["step"], vl]]
                    self.state["history"] = hist[-80:]
                    if self.state["best_val"] is None or vl < self.state["best_val"]:
                        self.state["best_val"] = vl
                if self.state["step"] >= next_save:
                    next_save = self.state["step"] + SAVE_EVERY
                    try:
                        self._save()
                    except OSError as e:
                        self.state["note"] = f"could not save: {e}"
            self.state["mem_mb"] = rss_mb()
            self._stop.wait(PAUSE)                      # hand the CPU back to Flask

    def start(self):
        if (self.model is None and self.music is None) or (self.thread and self.thread.is_alive()):
            return
        self._stop.clear()
        self.thread = threading.Thread(target=self._loop, daemon=True, name="mycoder-train")
        self.thread.start()

    def stop(self):
        self._stop.set()
        self.state["status"] = "paused"

# --- web app

PAGE = """<!doctype html>
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Nova</title>
<style>
:root{color-scheme:dark}
body{background:#101820;color:#d8e2ec;font:14px/1.6 ui-monospace,Menlo,monospace;
max-width:840px;margin:36px auto;padding:0 16px}
h1{font-size:17px;margin:0 0 3px}
.sub{color:#6d8296;font-size:12px;margin:0 0 18px}
.panel{border:1px solid #24323f;border-radius:3px;padding:14px;margin-bottom:18px;
background:#0c141c}
.stats{display:flex;gap:20px;flex-wrap:wrap;font-size:12px}
.stat b{display:block;font-size:16px;color:#7fd1a8;font-weight:600}
.stat span{color:#6d8296}
#spark{width:100%;height:54px;margin-top:12px}
textarea{width:100%;height:170px;background:#0a121a;color:#d8e2ec;border:1px solid #24323f;
padding:12px;font:inherit;border-radius:3px;resize:vertical}
.row{display:flex;gap:12px;align-items:center;margin-top:10px;flex-wrap:wrap}
label{font-size:12px;color:#6d8296}
input[type=range]{width:100px;vertical-align:middle}
button,a.btn{background:#2f7d76;color:#eefbf8;border:0;padding:9px 15px;border-radius:3px;
font:inherit;cursor:pointer;text-decoration:none;display:inline-block}
.ghost{background:none;border:1px solid #24323f;color:#6d8296}
#go{margin-left:auto}
pre{background:#0a121a;border:1px solid #24323f;border-left:3px solid #2f7d76;padding:14px;
white-space:pre-wrap;border-radius:3px;margin-top:16px}
.new{color:#7fd1a8}
.note{color:#c2915a;font-size:12px}
</style>
<h1>Nova <span id="mname" style="color:#6d8296;font-size:12px">1.7</span></h1>
<p class="sub">One file. Rio makes music, Milo does words.</p>
<div class="panel">
<div class="row" style="margin-top:0">
<select id="model" style="background:#0a121a;color:#d8e2ec;border:1px solid #24323f;
padding:8px;border-radius:3px;font:inherit"></select>
<input id="acct" placeholder="account" style="width:110px;background:#0a121a;
color:#d8e2ec;border:1px solid #24323f;padding:8px;border-radius:3px;font:inherit">
<span class="note" id="mblurb"></span>
</div>
<div id="log" style="margin:12px 0;max-height:340px;overflow-y:auto"></div>
<div class="row" style="margin-top:0">
<input id="msg" placeholder="make me a funk beat at 140 bpm"
style="flex:1;min-width:200px;background:#0a121a;color:#d8e2ec;border:1px solid #24323f;
padding:10px;font:inherit;border-radius:3px">
<button id="send">Send</button>
</div>
</div>
<div class="panel">
<div class="stats">
<div class="stat"><b id="s-step">–</b><span>steps</span></div>
<div class="stat"><b id="s-loss">–</b><span>val loss</span></div>
<div class="stat"><b id="s-best">–</b><span>best</span></div>
<div class="stat"><b id="s-params">–</b><span>params</span></div>
<div class="stat"><b id="s-mem">–</b><span>memory</span></div>
<div class="stat"><b id="s-status">–</b><span>status</span></div>
</div>
<svg id="spark" viewBox="0 0 400 54" preserveAspectRatio="none"></svg>
<div class="row">
<button class="ghost" id="pause">Pause training</button>
<a class="btn ghost" href="/export">Download weights</a>
<span class="note" id="note"></span>
</div>
</div>
<textarea id="p" placeholder="def evaluate_board(board):"></textarea>
<div class="row">
<label>length <input type="range" id="t" min="30" max="300" value="120"></label>
<label>temp <input type="range" id="temp" min="1" max="15" value="7"></label>
<button id="go">Continue</button>
</div>
<pre id="out">Below about 1.5 val loss it starts producing
real-looking code. Watch the curve.</pre>

<div id="results" style="margin-top:12px;font-size:12.5px"></div>
</div>
<script>
const $=id=>document.getElementById(id);
let paused=false;
function draw(h){
const svg=$('spark');
svg.innerHTML='';
if (h.length<2) return;
const ys=h.map(p=>p[1]),lo=Math.min(...ys),hi=Math.max(...ys),span=(hi - lo) || 1;
const pts=h.map((p,i)=>`${
(i/(h.length-1)*400).toFixed(1)},`+`${
(50-((p[1]-lo)/span)*46).toFixed(1)}`).join(' ');
const l=document.createElementNS('http://www.w3.org/2000/svg','polyline');
l.setAttribute('points',pts);
l.setAttribute('fill','none');
l.setAttribute('stroke','#2f7d76');
l.setAttribute('stroke-width','1.5');
svg.appendChild(l);
}async function poll(){
try{
const s=await (await fetch('/status')).json();
$('s-step').textContent=s.step.toLocaleString();
$('s-loss').textContent=s.val_loss ?? '–';
$('s-best').textContent=s.best_val ?? '–';
$('s-params').textContent=(s.params/1000).toFixed(0)+'K';
$('s-mem').textContent=s.mem_mb+' MB';
$('s-status').textContent=s.status;
$('note').textContent=s.note || '';
if (s.music){
$('m-step').textContent=s.music.step.toLocaleString();
$('m-loss').textContent=s.music.loss ?? '–';
$('m-style').textContent=s.music.style;
}draw(s.history);
}catch (e){
}}poll();
setInterval(poll,4000);
$('pause').onclick=async ()=>{
paused=!paused;
await fetch(paused ? '/pause':'/resume',{
method:'POST'});
$('pause').textContent=paused ? 'Resume training':'Pause training';
poll();
};
const log=$('log'),msg=$('msg');
function bb(who,text,cls){
const d=document.createElement('div');
d.style.margin='9px 0';
const col=who==='you' ? '#c2915a':'#7fd1a8';
d.innerHTML=`<span style="color:${
col};
font-size:11px">${
who}</span>`;
const p=document.createElement('div');
p.style.whiteSpace='pre-wrap';
if (cls){
p.style.background='#0a121a';
p.style.border='1px solid #24323f';
p.style.padding='10px';
p.style.borderRadius='3px';
p.style.marginTop='4px';
}p.textContent=text;
d.appendChild(p);
log.appendChild(d);
log.scrollTop=log.scrollHeight;
return d;
}async function lm(){
const d=await (await fetch('/models')).json();
const sel=$('model');
for (const [id,m] of Object.entries(d.models)){
const o=document.createElement('option');
o.value=id;
o.textContent=m.name;
if (id===d.default) o.selected=true;
sel.appendChild(o);
}const upd=()=>{
$('mblurb').textContent=d.models[sel.value].blurb;
$('mname').textContent=d.models[sel.value].name;
};
sel.onchange=upd;
upd();
}lm();
async function send(){
const text=msg.value.trim();
if (!text) return;
bb('you',text);
msg.value='';
const th=bb('milo','…');
try{
const ch=$('model').value;
const r=await (await fetch('/chat',{
method:'POST',headers:{
'Content-Type':'application/json'},body:JSON.stringify({
message:text,model:ch})})).json();
th.remove();
bb('milo',r.reply);
if (r.code) {
bb('milo',r.code,true);
const rb=document.createElement('button');
rb.textContent='Run it';
rb.style.cssText='margin:6px 0';
const ob=document.createElement('pre');
ob.style.cssText='display:none;border-left:3px solid #7fd1a8;margin-top:6px';
rb.onclick=async()=>{
rb.disabled=true;rb.textContent='running...';
try{
const d=await(await fetch('/run',{method:'POST',
headers:{'Content-Type':'application/json'},body:JSON.stringify({session:'web'})})).json();
ob.style.display='block';
ob.textContent=d.error?('error: '+d.error):(d.output||'(ran, printed nothing)');
ob.style.borderLeftColor=d.error?'#c7305a':'#7fd1a8';
}catch(e){ob.style.display='block';ob.textContent='failed: '+e.message;}
rb.disabled=false;rb.textContent='Run it again';};
log.appendChild(rb);log.appendChild(ob);log.scrollTop=log.scrollHeight;}
if (r.audio){
const a=document.createElement('audio');
a.controls=true;
a.src=r.audio;
a.style.width='100%';
a.style.marginTop='6px';
log.appendChild(a);
a.play().catch(()=>{
});
log.scrollTop=log.scrollHeight;
}if (r.links && r.links.length){
const d=document.createElement('div');
d.style.margin='6px 0 0';
d.innerHTML=r.links.map(l=>`<div style="margin-bottom:7px">`+`<a href="${
l.url}" target="_blank" style="color:#7fd1a8">${
l.title}</a><br>`+`<span style="color:#6d8296;font-size:12px">${
l.snippet||''}</span></div>`).join('');
log.appendChild(d);
log.scrollTop=log.scrollHeight;
}}catch (e){
th.remove();
bb('milo','Failed:'+e.message);
}}$('send').onclick=send;
msg.addEventListener('keydown',e=>{
if (e.key==='Enter') send();});
const pl=$('pl'),nb=$('notes');
const rs=$('rs');
$('t-music').onchange=pt;
const go=$('go'),out=$('out');
go.onclick=async ()=>{
const pr=$('p').value;
if (!pr.trim()) return;
go.disabled=true;
out.textContent='writing…';
try{
const r=await fetch('/complete',{
method:'POST',headers:{
'Content-Type':'application/json'},body:JSON.stringify({
pr,tokens:+$('t').value,temp:+$('temp').value/10})});
const d=await r.json();
if (d.error) out.textContent=d.error;
else{
out.textContent=pr;
const s=document.createElement('span');
s.className='new';
s.textContent=d.text;
out.appendChild(s);
}}catch (e){
out.textContent='Could not reach the model:'+e.message;
}go.disabled=false;
};
</script>"""

# --- milo: chat
# Routes what you ask to something it can do. No language model writes the
# replies; the work is real.

BASE = ["status", "help"]
MUSIC = BASE + ["music", "midi", "arrange"]  # Rio is the music model and
DEEP = MUSIC + ["styles", "long", "stereo"]  # gets the whole kit at base
TALK = BASE + ["search", "code"]            # Milo does words, not music
MODELS = {
    "rio-1.6":      {"name": "Rio 1.6", "can": MUSIC,
                     "blurb": "6 genres, arrangement, MIDI, follow-ups."},
    "rio-1.6-pro":  {"name": "Rio 1.6 Pro", "can": DEEP,
                     "blurb": "6 genres, stereo, folk, 32 bars."},
    "milo-1.8":     {"name": "Milo 1.8", "can": TALK + ["explain"],
                     "blurb": "Code that explains itself, plus search."},
    "nova-iris":    {"name": "Nova Iris", "warm": True, "paid": True,
                     "can": TALK + ["open", "compose", "explain", "deep"],
                     "blurb": "Milo 1.8 Pro, but it checks its work. $7/mo."},
    "milo-1.8-pro": {"name": "Milo 1.8 Pro", "warm": True,
                     "can": TALK + ["open", "compose", "explain"],
                     "blurb": "Understands loose phrasing, typos included."},
}
DEFAULT_MODEL = env("MYCODER_MODEL", "rio-1.6")
ADMINS = set(env("NOVA_ADMINS", "editornova").split(","))
SUBSCRIBERS = set(w for w in env("NOVA_PAID", "").split(",") if w)


def may_use(model, account=""):
    """Paid tiers need a subscription, admins skip it. Honour system, not
    security: anyone with this file can edit the list."""
    if not MODELS.get(model, {}).get("paid"):
        return True, ""
    who = (account or env("NOVA_ACCOUNT", "")).strip().lower()
    if who in ADMINS:
        return True, f"admin ({who})"
    if who in SUBSCRIBERS:
        return True, "subscriber"
    return False, ("Nova Iris is $7/month. Sign in as a subscriber, or use "
                   "Milo 1.8 Pro — free, and nearly as good.")

def _num(pattern, text, default=None):
    import re
    m_ = re.search(pattern, text, re.I)
    return int(m_.group(1)) if m_ else default

# Code that runs and explains itself: every snippet carries plain-English
# comments the explain intent reads back.

# One blob split on @@.
_SNIPPET_SRC = """
@@reverse|backwards
# A string is letters in a row, like beads on a thread. [::-1] means
# walk the row backwards: same beads, opposite order.
# Try it: reverse("funk") gives "knuf"
def reverse(text):
    return text[::-1]
@@fizzbuzz
# Count up. For each number: does 3 divide evenly? does 5? (% is the
# remainder, so % 3 == 0 means it fits.) Say Fizz, Buzz, both, or the
# number. Try it: fizzbuzz(5) -> 1 2 Fizz 4 Buzz
def fizzbuzz(n=100):
    for i in range(1, n + 1):
        out = ("Fizz" if i % 3 == 0 else "") + ("Buzz" if i % 5 == 0 else "")
        print(out or i)
@@fibonacci|fib
# Each number is the two before it added up: 0, 1, 1, 2, 3, 5...
# a, b = b, a + b shuffles them along one place each loop.
# Try it: fib(6) -> [0, 1, 1, 2, 3, 5]
def fib(n):
    a, b, out = 0, 1, []
    for _ in range(n):
        out.append(a)
        a, b = b, a + b
    return out
@@prime
# A prime divides only by 1 and itself, so try every number below it.
# We stop at the square root because factors come in pairs.
# Try it: is_prime(7) True, is_prime(9) False
def is_prime(n):
    if n < 2:
        return False
    for d in range(2, int(n ** 0.5) + 1):
        if n % d == 0:
            return False
    return True
@@count word|word count
# A dict is labelled boxes: the word labels it, the count is inside.
# counts.get(word, 0) means "what's there, or 0 if it's new".
# Try it: count_words("a b a") -> {a: 2, b: 1}
def count_words(text):
    counts = {}
    for word in text.lower().split():
        counts[word] = counts.get(word, 0) + 1
    return counts
@@binary search|bisect
# Like a dictionary: open the middle, keep the half it's in, repeat.
# Halving each guess means 1000 items take ~10 tries. Must be sorted.
# Try: binary_search([1, 3, 5], 5) -> 2
def binary_search(items, target):
    lo, hi = 0, len(items) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if items[mid] == target:
            return mid
        if items[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
@@sort|dictionary
# Sorting normally goes by the label; here we sort by what's inside.
# key= says which bit to compare — kv[1] is the value, not the label.
# Try it: sort_by_value({a: 1, b: 9}) puts b first, because 9 > 1
def sort_by_value(d, biggest_first=True):
    return dict(sorted(d.items(), key=lambda kv: kv[1], reverse=biggest_first))
@@read a file|read file|open a file
# Opening a file is like a drawer: you must close it. `with` closes it
# for you when the block ends, even if something breaks.
def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [line.rstrip() for line in f]
@@class
# A class is a cookie cutter; each thing you make is a cookie.
# __init__ runs when you make one, self is "this particular one",
# __repr__ is how it prints. Try: Thing("kick") -> Thing('kick')
class Thing:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"Thing({self.name!r})"
@@average|mean
# The total shared out evenly. sum() adds them, len() counts them.
# The if guards an empty list, since dividing by zero crashes.
# Try: average([2, 4]) -> 3.0
def average(numbers):
    return sum(numbers) / len(numbers) if numbers else 0.0
@@chess|board|evaluate|piece
# Chess engines guess who's winning by counting material: pawn 1,
# knight/bishop 3, rook 5, queen 9. White is uppercase so it adds,
# black subtracts. Try it: evaluate(["Q", "p"]) -> 8
VALUES = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9, "k": 0}

def evaluate(board):
    # material score; uppercase white, lowercase black
    score = 0
    for piece in board:
        if piece.lower() in VALUES:
            score += VALUES[piece.lower()] * (1 if piece.isupper() else -1)
    return score
"""

SNIPPETS = []
for _chunk in _SNIPPET_SRC.strip().split("@@"):
    _head, _, _body = _chunk.strip("\n").partition("\n")
    if _body.strip():
        SNIPPETS.append((_head.split("|"), _body.rstrip()))

def write_code(request, adapt=False):
    """-> (code, matched). adapt=True renames it to what you asked for."""
    import re as _re
    t = request.lower()
    want = _re.search(r"(?:called|named)\s+([a-z_][a-z0-9_]*)", t)
    # A vague word counts only when the request is vague too.
    weak = {"class", "sort", "board", "find", "make", "read", "swap"}
    for keys, code in SNIPPETS:
        hits = [k for k in keys if k in t]
        if hits and (set(hits) - weak or len(t.split()) <= 3):
            if adapt and want:
                first = _re.search(r"def ([a-z_][a-z0-9_]*)", code)
                if first:
                    code = code.replace(first.group(1), want.group(1))
            return code, True
    m_ = _re.search(r"(?:function|def)\s+(?:called\s+)?([a-z_][a-z0-9_]*)", t)
    if m_:
        name = m_.group(1)
    else:                                  # name it after what you asked for
        skip = {"write", "make", "a", "an", "the", "that", "code", "for", "to",
                "function", "python", "me", "with", "and", "of", "in", "some"}
        words = [w for w in _re.findall(r"[a-z]+", t) if w not in skip][:3]
        name = "_".join(words) or "do_thing"
    return (f"# I don't know this one yet, so here's the shape to fill in.\n"
            f"# def names it, {name} is what you'll call it, and value is\n"
            f"# whatever you feed it. Change the middle line to do the work,\n"
            f"# then return hands your answer back out.\n"
            f"# Try it: {name}(5) gives 5 back, until you change it.\n"
            f"def {name}(value):\n"
            f'    """{request.strip()[:60]}"""\n'
            f"    result = value\n"
            f"    return result"), False

_re_ask = __import__("re").compile(
    r"^(convert|merge|remove|find|check|calculate|count|reverse|swap|turn|"
    r"make(?! it)|build|create|parse|fetch|sort|split|join)\b")
_re_open = __import__("re").compile(r"^(open|read)\s+(the\s+)?(\d+|first|second)")

# Understanding without a language model: each intent owns a bag of words,
# and the message scores against all of them at once.
STOP = set("a an the i you it that this to for of and is are my me we us can do"
           " with some something please just".split())
# Filler words like "me" swallowed sentences, so they go first.
WORDS = {
    "greet":    "hi hey hello yo sup yow howdy morning hows going",
    "thanks":   "thanks thank cheers appreciate nice sick lovely great"
                " awesome wicked",
    "identity": "who yourself name model called version robot alive",
    "help":     "help options commands examples able handle capable stuck",
    "status":   "status trained parameters loss score built version",
    "midi":     "midi mid daw logic ableton reaper cubase stems export",
    "open":     "open link result page article site url first second third",
    "explain":  "explain simpler clearer break walk through meaning means"
                " mean understand confused lost why",
    "search":   "search google lookup news latest happening won winner"
                " when where wiki info facts",
    "music":    "beat track song loop tune banger rhythm groove riff melody"
                " bassline drum drums percussion bpm bars tempo funk trap"
                " house dnb lofi reggaeton hear listen play slow fast heavy"
                " dark chill",
    "code":     "code function class script program routine snippet method"
                " algorithm reverse flip backwards sort order count tally"
                " average prime factors fibonacci fizzbuzz merge combine"
                " remove duplicate tidy convert calculate palindrome list"
                " dict string number numbers file json",
}
BAGS = {k: set(v.split()) - STOP for k, v in WORDS.items()}
SPREAD = {}
for _bag in BAGS.values():
    for _w in _bag:
        SPREAD[_w] = SPREAD.get(_w, 0) + 1
VOCAB = sorted(SPREAD)


def detect_intent(text):
    """Score against each intent's bag, allowing for typos."""
    import difflib
    import re as _re
    t = text.lower().strip()
    if not t:
        return "empty"
    raw = [w for w in _re.findall(r"[a-z']+", t)]
    words = set()
    for w in raw:
        if w in STOP:
            continue
        if w in SPREAD:
            words.add(w)
        elif len(w) > 3:                       # "somethign" -> "something"
            near = difflib.get_close_matches(w, VOCAB, 1, 0.78)
            words.add(near[0] if near else w)
        else:
            words.add({"u": "you", "r": "are", "ur": "your",
                       "wut": "what", "pls": "please"}.get(w, w))

    if any(t.startswith(k) for keys, _ in SNIPPETS for k in keys):
        return "code"
    if _re_open.match(t):
        return "open"
    scores = {k: sum(1.0 / SPREAD[w] for w in words & bag)
              for k, bag in BAGS.items()}
    # phrases beat single words: "who won" is a search, not about me
    for phrase, intent in (("search for", "search"), ("look up", "search"),
                           ("pull up", "search"), ("who won", "search"),
                           ("how good", "status"), ("how were you", "status"),
                           ("what version", "status"), ("who are you", "identity"),
                           ("what are you", "identity"), ("r u", "identity"),
                           ("walk me", "explain"), ("say it", "explain"),
                           ("what does", "explain"), ("what can", "help"),
                           ("put together", "music"), ("logic pro", "midi"),
                           ("ableton", "midi")):
        if phrase in t:
            scores[intent] += 2.5
    if _re_ask.match(t) and not scores["music"]:
        scores["code"] += 2
    if words & {"average", "flip", "backwards", "divisible", "factors"}:
        scores["code"] += 1.5
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    top, second = ranked[0], ranked[1]
    if top[1] < 0.5:
        return "unknown"
    if top[1] - second[1] < 0.35:              # too close to call: ask instead
        return f"unsure:{top[0]}:{second[0]}"
    return top[0]


# Pro's warmth is phrasing and memory, not understanding.
WARM = {
    "greet": ["Hey{name}. What are we making?", "Yo{name}.",
              "Hey{name}. Beat or code?"],
    "ack":   ["On it.", "Sure thing.", "Alright.", "Got it"],
    "thanks": ["Anytime.", "Np.", "Glad it helped."],
    "again": ["Another coming up.", "Take two.", "Running it again."],
    "sorry": ["Outside what I do.", "Can't do that, sorry."],
}


class ChatSession:
    """Remembers enough for follow-ups."""

    def __init__(self, account=""):
        self.account = account
        self.name = None
        self.last = {}
        self.turns = 0

    def pick(self, kind, **kw):
        seed = np.random.default_rng(self.turns * 7 + len(kind))
        line = WARM[kind][int(seed.integers(0, len(WARM[kind])))]
        return line.format(name=f" {self.name}" if self.name else "", **kw)


SESSIONS = {}


def get_session(key="default", account=""):
    ses = SESSIONS.setdefault(key, ChatSession(account))
    if account:
        ses.account = account
    return ses


def run_code(code, timeout=5):
    """Run generated code in a child process. Only ever code Nova made for
    this session, never browser text, with a timeout."""
    import re as _re
    import subprocess
    import sys as _sys
    demo = _re.search(r"Try(?: it)?:\s*([A-Za-z_]\w*\([^)]*\))", code)
    script = code
    if demo:
        call = demo.group(1)
        script += f"\n\n_r = {call}\nif _r is not None:\n    print(_r)"
    elif "print(" not in code:
        return "", "Nothing to show — this one has no example to run."
    try:
        p = subprocess.run([_sys.executable, "-c", script], capture_output=True,
                           text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "", f"Stopped after {timeout}s — it never finished."
    except Exception as e:
        return "", f"Couldn't run it ({type(e).__name__})."
    return p.stdout.strip(), p.stderr.strip().split("\n")[-1] if p.stderr else ""

def scores(spec):
    """0-10 measured; nothing hits 10."""
    c = spec["can"]
    return {"music": (9 if "stereo" in c else 6) if "music" in c else 0,
            "code": (9 if "compose" in c else 7) if "code" in c else 0,
            "chat": 4 + 3 * bool(spec.get("warm")),
            "search": 8 if "search" in c else 0}

# Pro's warmth is phrasing and memory, not understanding.
WARM = {
    "greet": ["Hey{name}. What are we making?", "Yo{name}.",
              "Hey{name}. Beat or code?"],
    "ack":   ["On it.", "Sure thing.", "Alright.", "Got it"],
    "thanks": ["Anytime.", "Np.", "Glad it helped."],
    "again": ["Another coming up.", "Take two.", "Running it again."],
    "sorry": ["Outside what I do.", "Can't do that, sorry."],
}


def milo_reply(text, trainer, model=DEFAULT_MODEL, session=None):
    """-> dict: reply plus audio/links/code/midi when relevant."""
    spec = MODELS.get(model, MODELS[DEFAULT_MODEL])
    allowed, why = may_use(model, getattr(session, "account", ""))
    if not allowed:
        return {"intent": "locked", "model": spec["name"], "reply": why}
    warm = spec.get("warm")
    ses = session or get_session()
    ses.turns += 1
    low = text.lower().strip()

    import re as _re
    m_ = _re.search(r"(?:i'?m|my name is|call me)\s+([a-z]{2,15})", low)
    if m_ and warm and m_.group(1) not in ("looking", "trying", "going", "not"):
        ses.name = m_.group(1).capitalize()
        return {"intent": "name", "model": spec["name"],
                "reply": f"Good to meet you, {ses.name}. What are we making?"}

    intent = detect_intent(text)
    if intent.startswith("unsure:"):
        _, a, b = intent.split(":")
        names = {"music": "a beat", "code": "some code", "search": "a search",
                 "explain": "an explanation", "midi": "the MIDI",
                 "status": "my stats", "identity": "who I am", "help": "help",
                 "open": "that link", "greet": "a hello"}
        return {"intent": "unsure", "model": spec["name"],
                "reply": f"Did you want {names.get(a, a)} or "
                         f"{names.get(b, b)} — which one?"}

    # follow-ups only make sense with memory, so they're a Pro thing
    if (warm or "music" in spec["can"]) and ses.last:
        if any(w in low for w in ("heavier", "harder", "darker", "chill")):
            intent = "music"
        if any(w in low for w in ("again", "another", "one more", "different one")):
            # keep the settings from last time, or "faster" then "again" undoes itself
            intent = "music"
            text += f" {ses.last.get('bpm', FUNK_BPM)} bpm {ses.last.get('bars', 8)} bars"
        elif any(w in low for w in ("faster", "quicker", "speed it up")):
            intent = "music"; text += f" {min(180, ses.last.get('bpm', FUNK_BPM) + 15)} bpm"
        elif any(w in low for w in ("slower", "slow it down", "chill")):
            intent = "music"; text += f" {max(80, ses.last.get('bpm', FUNK_BPM) - 15)} bpm"
        elif any(w in low for w in ("longer", "extend")):
            # carry the tempo too, or "faster" then "longer" silently resets it
            intent = "music"
            text += (f" {min(16, ses.last.get('bars', 8) + 4)} bars"
                     f" {ses.last.get('bpm', FUNK_BPM)} bpm")

    if warm and any(w in low for w in ("thanks", "thank you", "nice", "sick", "love it")):
        return {"intent": "thanks", "model": spec["name"], "reply": ses.pick("thanks")}

    out = {"intent": intent, "model": spec["name"]}

    if intent not in spec["can"] and intent in (
            "code", "search", "midi", "open", "music", "explain"):
        others = [m["name"] for k, m in MODELS.items() if intent in m["can"]]
        lead = ses.pick("sorry") if warm else f"{spec['name']} doesn't do that."
        out["reply"] = (f"{lead} {' or '.join(others)} can — switch model above."
                        if others else lead)
        return out

    if intent == "empty":
        out["reply"] = "Say something."

    elif intent == "greet":
        out["reply"] = (ses.pick("greet") if warm
                        else f"Hey. I'm {spec['name']}. Ask me for a beat, or type help.")

    elif intent == "identity":
        others = ", ".join(m["name"] for k, m in MODELS.items() if k != model)
        out["reply"] = (
            f"I'm {spec['name']}. {spec['blurb']} Also here: {others}.\n"
            f"Straight with you: no language model writes these replies. I match "
            f"what you ask against what I can do, then do it. The friendliness "
            f"is hand-written, not thought up.")

    elif intent == "help":
        lines = ["Try:", "  make a trap beat at 140 bpm",
                 "  make a 16 bar house loop", "  midi", "  status"]
        if "search" in spec["can"]:
            lines += ["  search for funk history"]
        if "code" in spec["can"]:
            lines.append("  write a function that reverses a string")
        if warm:
            lines += ["", "Follow-ups: 'again', 'faster', 'longer'."]
        out["reply"] = "\n".join(lines)

    elif intent == "status":
        ms = (trainer.music.state if trainer.music else {})
        sc = scores(spec)
        out["scores"] = sc
        out["reply"] = (f"{spec['name']}  "
                        + "  ".join(f"{k} {v}/10" for k, v in sc.items())
                        + f"\n  music: {ms.get('params', 0):,} params, "
                        f"style {ms.get('style')}, from {ms.get('source')}"
                        f"\n  funk hooks come from the generator, which beat "
                        f"the model 98% to 95% on staying in key")

    elif intent == "midi":
        if not ses.last.get("tokens"):
            out["reply"] = "Make something first."
            return out
        import base64
        mid = tokens_to_midi(ses.last["tokens"], bpm=ses.last.get("bpm", FUNK_BPM))
        out["midi"] = "data:audio/midi;base64," + base64.b64encode(mid).decode()
        out["reply"] = f"MIDI, {len(mid):,} bytes — opens in any DAW."

    elif intent == "explain":
        code = ses.last.get("code")
        if not code:
            out["reply"] = "Ask me for some code first, then I'll explain it."
            return out
        why = [l[2:] for l in code.split("\n") if l.startswith("# ")]
        body = [l for l in code.split("\n") if l.strip()
                and not l.strip().startswith("#")]
        walk = []
        for line in body:
            bit = line.strip()
            if bit.startswith("def "):
                walk.append(f"  {bit}  <- names it, and lists what you feed in")
            elif bit.startswith("class "):
                walk.append(f"  {bit}  <- the template everything is made from")
            elif bit.startswith("for "):
                walk.append(f"  {bit}  <- do the next bit once for each item")
            elif bit.startswith("while "):
                walk.append(f"  {bit}  <- keep going while that stays true")
            elif bit.startswith("if ") or bit.startswith("elif "):
                walk.append(f"  {bit}  <- only do this when that's true")
            elif bit.startswith("return"):
                walk.append(f"  {bit}  <- hand the answer back")
            elif bit.startswith("with "):
                walk.append(f"  {bit}  <- borrow it, and tidy up afterwards")
            elif "=" in bit and not bit.startswith(("print", "raise")):
                walk.append(f"  {bit}  <- remember this for later")
            else:
                walk.append(f"  {bit}")
        out["reply"] = ("\n".join(why) + "\n\nLine by line:\n" + "\n".join(walk)
                        + "\n\nPaste it into a file and call it to try it.")
        out["code"] = code

    elif intent == "open":
        idx = (_num(r"(\d+)", text, 1) or 1) - 1
        links = ses.last.get("links") or []
        if not links:
            out["reply"] = "Search first."
            return out
        if idx >= len(links):
            out["reply"] = f"Only {len(links)} results."
            return out
        try:
            body = fetch_page(links[idx]["url"])
        except Exception as e:
            out["reply"] = f"Couldn't open that ({type(e).__name__})."
            return out
        out["reply"] = f"{links[idx]['title']}\n\n{body[:700]}…"

    elif intent == "music":
        if trainer.music is None:
            out["reply"] = "Music model is off."
            return out
        bpm = _num(r"(\d{2,3})\s*bpm", text)
        bars = _num(r"(\d{1,2})\s*bars?", text, 8)
        note = ""
        style = ("real" if any(w in text.lower() for w in ("folk", "melodic", "tune"))
                 else "funk")
        genre = next((k for k in GENRES if k in text.lower()), "funk")
        cap = 32 if "long" in spec["can"] else 16
        if bars > cap:
            bars, note = cap, f" (capped at {cap}; Pro does 32.)"
        if style == "real" and "styles" not in spec["can"]:
            style, note = "funk", f" ({spec['name']}: funk only.)"
        old = trainer.music.style
        if style == "real" and trainer.music.model_missing:
            # folk needs its weights; don't quietly serve funk
            style, note = "funk", " (no music-real.npz, so funk)"
        try:
            if style != old:
                trainer.music.style = style
            wav, tokens = trainer.music.compose(bars=max(2, bars), bpm=bpm,
                                                arrange="arrange" in spec["can"],
                                                stereo="stereo" in spec["can"],
                                                genre=genre)
        finally:
            trainer.music.style = old
        import base64
        out["audio"] = "data:audio/wav;base64," + base64.b64encode(wav).decode()
        out["notes"] = tokens
        shown = bpm or GENRES[genre]["bpm"]        # not the funk default
        ses.last.update(bars=bars, bpm=shown, style=style, tokens=tokens)
        lead = (ses.pick("again") if "again" in text.lower() else ses.pick("ack")) if warm else ""
        kind = genre if style == "funk" else "folk melody"
        out["reply"] = (f"{lead} {bars} bars of {kind} at {shown} bpm.{note}"
                        if warm else
                        f"Here's {bars} bars of {kind} at {shown} bpm.{note}")

    elif intent == "search":
        q = text
        for prefix in ("search for", "search", "look up", "google", "find me info on"):
            if q.lower().startswith(prefix):
                q = q[len(prefix):]
                break
        try:
            results = web_search(q.strip(" ?:"), count=5)
        except Exception as e:
            out["reply"] = f"Couldn't reach the search engine ({type(e).__name__})."
            return out
        out["links"] = results
        ses.last["links"] = results
        engine = "Google" if (GOOGLE_KEY and GOOGLE_CX) else "DuckDuckGo"
        out["reply"] = (f"{len(results)} results for “{q.strip()}” via {engine}."
                        if results else "Nothing came back.")

    elif intent == "code":
        prompt = text
        for p in ("finish this code:", "write me a", "write a", "code for", "code:"):
            if p in prompt.lower():
                prompt = prompt[prompt.lower().index(p) + len(p):]
                break
        prompt = prompt.strip() or "def "
        code, real = write_code(prompt, adapt="compose" in spec["can"])
        out["code"] = code
        if "deep" in spec["can"]:
            # Iris runs it first, and says so if it breaks.
            good, err = run_code(out["code"])
            if err and "Nothing to show" not in err:
                out["checked"] = f"heads up, it errored: {err[:70]}"
            elif good:
                out["checked"] = f"checked it — output: {good.splitlines()[0][:50]}"
        ses.last["code"] = out["code"]
        why = [l[2:] for l in out["code"].split("\n") if l.startswith("# ")]
        if real:
            head = "Here you go, this runs."
            if out.get("checked"):
                head += "\n" + out["checked"]
            out["reply"] = (head + "\n" + "\n".join(why)
                            if why and "explain" in spec["can"] else head)
        else:
            out["reply"] = "No rule for that — here's a skeleton."

    else:
        out["reply"] = ("Didn't catch that. I do music, search, code, status "
                        "— type help.")
    return out

def build_app(trainer):
    from flask import Flask, jsonify, render_template_string, request, send_file
    app = Flask(__name__)

    @app.route("/")
    def index():
        if AUTOTRAIN:
            trainer.start()                             # a visit wakes it back up
        return render_template_string(PAGE)

    @app.route("/status")
    def status():
        trainer.state["mem_mb"] = rss_mb()
        return jsonify(trainer.state)

    @app.route("/complete", methods=["POST"])
    def complete():
        body = request.get_json(silent=True) or {}
        prompt = body.get("prompt", "")
        if not prompt.strip():
            return jsonify(error="Nothing to continue — type some code first."), 400
        if trainer.model is None:
            return jsonify(error=trainer.state.get("note") or "No model yet."), 503
        return jsonify(text=trainer.complete(prompt,
                                             tokens=min(int(body.get("tokens", 120)), 300),
                                             temp=float(body.get("temp", 0.8))),
                       step=trainer.state["step"])

    @app.route("/export")
    def export():
        """Download the weights, commit the file, point MYCODER_CKPT_URL at it,
        and the next deploy carries on instead of starting over."""
        if trainer.model is None:
            return jsonify(error="No model to export."), 503
        buf = io.BytesIO()
        with trainer.lock:
            trainer.model.save(buf, extra={"step": trainer.state["step"],
                                           "val": trainer.state["best_val"]})
        buf.seek(0)
        return send_file(buf, mimetype="application/octet-stream", as_attachment=True,
                         download_name=f"weights-step{trainer.state['step']}.npz")

    def _dataurl(png):
        import base64
        return "data:image/png;base64," + base64.b64encode(png).decode()

    @app.route("/chat", methods=["POST"])
    def chat():
        body = request.get_json(silent=True) or {}
        return jsonify(milo_reply(body.get("message", ""), trainer,
                                  body.get("model", DEFAULT_MODEL),
                                  get_session(body.get("session", "web"),
                                              body.get("account", ""))))

    @app.route("/run", methods=["POST"])
    def run():
        body = request.get_json(silent=True) or {}
        ses = get_session(body.get("session", "web"))
        code = ses.last.get("code")
        if not code:
            return jsonify(error="Ask for some code first."), 400
        stdout, err = run_code(code)
        return jsonify(output=stdout, error=err)

    @app.route("/models")
    def models():
        out = {k: dict(v, scores=scores(v)) for k, v in MODELS.items()}
        return jsonify(models=out, default=DEFAULT_MODEL)

    @app.route("/search", methods=["POST"])
    def search():
        body = request.get_json(silent=True) or {}
        q = (body.get("query") or "").strip()
        if not q:
            return jsonify(error="Nothing to search for."), 400
        try:
            return jsonify(results=web_search(q, count=int(body.get("count", 5))))
        except Exception as e:
            return jsonify(error=f"Search failed ({type(e).__name__})."), 502

    @app.route("/targets", methods=["POST"])
    def targets():
        body = request.get_json(silent=True) or {}
        picked = {t for t in body.get("targets", []) if t in ("code", "music")}
        trainer.targets = picked
        trainer.state["targets"] = sorted(picked)
        return jsonify(ok=True, targets=sorted(picked))

    @app.route("/pause", methods=["POST"])
    def pause():
        trainer.stop(); return jsonify(ok=True)

    @app.route("/resume", methods=["POST"])
    def resume():
        trainer.start(); return jsonify(ok=True)

    @app.route("/healthz")
    def healthz():
        return jsonify(ok=True, step=trainer.state["step"], mem_mb=rss_mb())

    return app

# --- music
# The same transformer fed notes, not characters: a short sequence over a tiny
# vocabulary is what a small model can handle.

SR          = 22050                      # audio sample rate
MUSIC_STYLE = env("MYCODER_STYLE", "funk")           # funk | real | melodic
MUSIC_CKPT  = os.path.join(DATA_DIR, f"music-{env('MYCODER_STYLE', 'funk')}.npz")
MUSIC_VOCAB = os.path.join(DATA_DIR, f"music_vocab-{env('MYCODER_STYLE', 'funk')}.json")

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAJOR = [0, 2, 4, 5, 7, 9, 11]
MINOR = [0, 2, 3, 5, 7, 8, 10]
# scale degrees each chord is built from, as offsets into the scale
PROGRESSIONS = [[0, 4, 5, 3], [0, 5, 3, 4], [0, 3, 4, 4], [5, 3, 0, 4], [0, 0, 3, 4]]

def midi_name(midi):
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"

def name_midi(name):
    i = 2 if len(name) > 1 and name[1] == "#" else 1     # 'C#4' vs 'C4'
    pitch, octave = name[:i], name[i:]
    return NOTE_NAMES.index(pitch) + (int(octave) + 1) * 12

class NoteVocab:
    """Whitespace tokenizer over tokens like 'E4/4', 'R/2', '|'."""

    def __init__(self, tokens=None):
        self.itos = list(tokens or [])
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def build(self, text):
        seen = sorted(set(text.split()))
        self.itos = ["<pad>"] + seen
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        return self

    def encode(self, text):
        return [self.stoi[t] for t in text.split() if t in self.stoi]

    def coverage(self, text):
        """Fraction of a corpus this vocabulary covers."""
        toks = text.split()
        return sum(1 for t in toks if t in self.stoi) / max(1, len(toks))

    def decode(self, ids):
        return " ".join(self.itos[i] for i in ids if 0 <= i < len(self.itos))

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.itos, f)

    @classmethod
    def load(cls, path):
        return cls(json.load(open(path)))

    def __len__(self):
        return len(self.itos)

def make_melody(seed):
    """Pick a key, walk a progression."""
    r = np.random.default_rng(seed)
    root = int(r.integers(55, 68))                       # G3..G4
    scale = MAJOR if r.random() < 0.65 else MINOR
    prog = PROGRESSIONS[int(r.integers(0, len(PROGRESSIONS)))]
    bars, out = 8, []
    cur = root + 12
    for bar in range(bars):
        degree = prog[bar % len(prog)]
        chord = [scale[(degree + k) % 7] + 12 * ((degree + k) // 7) for k in (0, 2, 4)]
        beats = 16                                       # sixteenths per bar
        while beats > 0:
            dur = int(r.choice([2, 2, 4, 4, 4, 8], p=[.15, .15, .25, .2, .15, .10]))
            dur = min(dur, beats)
            if r.random() < 0.08:
                out.append(f"R/{dur}")
            else:
                if r.random() < 0.6:                     # land on a chord tone
                    target = root + chord[int(r.integers(0, 3))]
                else:
                    target = root + scale[int(r.integers(0, 7))]
                while target < cur - 7:
                    target += 12
                while target > cur + 7:
                    target -= 12
                cur = int(np.clip(target, 48, 84))
                out.append(f"{midi_name(cur)}/{dur}")
            beats -= dur
        out.append("|")
    return " ".join(out)

# --- brazilian funk
# Funk carioca: 16 steps, kicks on a tresillo, 808, claps, hook.

FUNK_BPM = 130
TRESILLO = [0, 3, 6, 10, 13]                 # the tamborzão kick skeleton
KICK_VARIANTS = [[0, 3, 6, 10, 13], [0, 3, 6, 8, 11, 14],   # tamborzão, mandelão
                 [0, 3, 7, 10, 13], [0, 4, 6, 10, 12, 14]]
CLAP_VARIANTS = [[4, 12], [4, 12, 14], [3, 7, 11, 15], [4, 10, 12]]

# Same engine, different numbers: kick/clap are 16-step patterns.
PHRYGIAN = [0, 1, 3, 5, 7, 8, 10]     # the b2 gives mandelão its darkness
GENRES = {
    "funk":      dict(bpm=130, kick=None, clap=None, hat=2, sub=.6, swing=0,
                      mode=None),
    "trap":      dict(bpm=140, kick=[0, 6, 10], clap=[8], hat=1, sub=.85,
                      swing=0, mode=PHRYGIAN),
    "house":     dict(bpm=124, kick=[0, 4, 8, 12], clap=[4, 12], hat=2,
                      sub=.35, swing=.12, mode=MINOR),
    "dnb":       dict(bpm=174, kick=[0, 10], clap=[4, 12], hat=2, sub=.5,
                      swing=0, mode=MINOR),
    "reggaeton": dict(bpm=95, kick=[0, 3, 8, 11], clap=[4, 12], hat=2,
                      sub=.55, swing=.08, mode=MINOR),
    "lofi":      dict(bpm=82, kick=[0, 10], clap=[8], hat=4, sub=.45,
                      swing=.18, mode=MINOR),
}
# Share of spectrum under 140 Hz; the renderer balances to it.
FUNK_LOW_TARGET = env("MYCODER_LOW_TARGET", 0.26, float)

# Funk rhythmic cells, in sixteenths.
CELLS = [
    [3, 3, 2], [3, 3, 4, 3, 3], [2, 2, 4], [4, 2, 2], [3, 5],
    [2, 2, 2, 2], [1, 1, 2, 4], [6, 2], [3, 1, 4],
]
CHOP_CELLS = [[1] * 4, [1, 1, 2], [1] * 6, [2, 1, 1]]   # montagem stutters

def _motif(r, root, mode, span=(0, 5)):
    """3-6 notes, small steps, one leap."""
    degrees = []
    d = int(r.integers(*span))
    for _ in range(int(r.integers(3, 7))):
        degrees.append(d)
        move = int(r.choice([-2, -1, -1, 0, 1, 1, 2], p=[.1, .25, .15, .1, .15, .15, .1]))
        d = int(np.clip(d + move, -2, 8))
    if r.random() < 0.35:
        degrees[int(r.integers(0, len(degrees)))] += 7
    out = []
    for deg in degrees:
        octave, step = divmod(deg, 7)
        out.append(root + mode[step] + 12 * octave)
    return out

def _events(r, pitches, cell, chop=False):
    """Cell + pitches -> tokens."""
    toks, i = [], 0
    for dur in cell:
        if not chop and r.random() < 0.12:
            toks.append(f"R/{dur}")
            continue
        p = pitches[i % len(pitches)] if not chop else pitches[0]
        i += 1
        toks.append(f"{midi_name(int(np.clip(p, 33, 88)))}/{dur}")
    return toks

def _bar(r, root, mode, motif, kind):
    """One bar of 16 sixteenths."""
    toks, left = [], 16
    while left > 0:
        if kind == "chop" and r.random() < 0.55:
            cell = CHOP_CELLS[int(r.integers(0, len(CHOP_CELLS)))]
            chop = True
        else:
            cell = CELLS[int(r.integers(0, len(CELLS)))]
            chop = False
        cell = [d for d in cell if d <= left] or [left]
        if sum(cell) > left:
            cell = cell[:1]
        toks += _events(r, motif, cell, chop)
        left -= sum(cell)
        if kind == "sparse" and left >= 4 and r.random() < 0.5:
            gap = int(r.choice([2, 4]))
            toks.append(f"R/{gap}")
            left -= gap
    return toks + ["|"]

def make_funk_hook(seed, genre="funk"):
    """Motif, answer, repeat."""
    r = np.random.default_rng(seed)
    flavour = str(r.choice(["mandelao", "montagem", "brega", "classic"]))
    g = GENRES.get(genre, GENRES["funk"])
    mode = g["mode"] or (PHRYGIAN if flavour == "mandelao" or r.random() < .25
                         else MINOR)
    root = int(r.integers(*((40, 47) if flavour in ("mandelao", "montagem")
                            else (47, 55))))

    call = _motif(r, root, mode, (0, 4))
    response = [p + int(r.choice([-2, 0, 2, 3])) for p in call]      # answer it
    kind = {"mandelao": "sparse", "montagem": "chop",
            "brega": "busy", "classic": "busy"}[flavour]

    bars = (_bar(r, root, mode, call, kind)
            + _bar(r, root, mode, call, kind if r.random() < 0.6 else "busy")
            + _bar(r, root, mode, response, kind)
            + _bar(r, root, mode, call, kind))
    return " ".join(bars * 2)                            # state the whole thing twice

def funk_corpus(n=12000, seed=0):
    return "\n".join(make_funk_hook(seed + i) for i in range(n))

# ---- drum and bass synthesis ----

def _env(n, decay):
    return np.exp(-np.arange(n, dtype=np.float32) / SR * decay)

def _highpass(x):
    """Cheap high pass: signal minus its smoothed self."""
    k = np.ones(9, dtype=np.float32) / 9
    return x - np.convolve(x, k, mode="same")

def _kick(dur=0.30):
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    freq = 42.0 + 95.0 * np.exp(-t * 32)                 # pitch drops fast: the thump
    phase = np.cumsum(2 * np.pi * freq / SR)
    body = np.sin(phase) * _env(n, 9)
    noise = np.random.default_rng(1).standard_normal(n).astype(np.float32)
    click = _highpass(noise) * _env(n, 400) * 0.25
    return np.tanh((body + click) * 1.7)                 # drive it

def _clap(seed=0, dur=0.26):
    n = int(dur * SR)
    r = np.random.default_rng(seed)
    noise = _highpass(r.standard_normal(n).astype(np.float32))
    out = np.zeros(n, dtype=np.float32)
    for k, off in enumerate((0.0, 0.011, 0.023)):        # three flams = a clap
        i = int(off * SR)
        out[i:] += (noise[:n - i] * _env(n - i, 55 if k < 2 else 22)) * (0.7 if k < 2 else 1.0)
    return out * 0.6

def _hat(seed=0, dur=0.055, open_=False):
    n = int(dur * (3 if open_ else 1) * SR)
    noise = _highpass(np.random.default_rng(seed).standard_normal(n).astype(np.float32))
    return noise * _env(n, 45 if open_ else 130) * 0.35

def _sub(midi, dur):
    """808 sub with a glide into the note."""
    n = max(1, int(dur * SR))
    t = np.arange(n, dtype=np.float32) / SR
    target = 440.0 * 2 ** ((midi - 69) / 12.0)
    freq = target * (1 + 0.5 * np.exp(-t * 45))          # glide
    phase = np.cumsum(2 * np.pi * freq / SR)
    return np.tanh(np.sin(phase) * 1.4) * _env(n, 3.2)

def _lead(midi, dur):
    n = max(1, int(dur * SR))
    t = np.arange(n, dtype=np.float32) / SR
    f = 440.0 * 2 ** ((midi - 69) / 12.0)
    tone = (np.sign(np.sin(2 * np.pi * f * t)) * 0.5
            + np.sign(np.sin(2 * np.pi * f * 1.005 * t)) * 0.5)   # slight detune
    envelope = np.ones(n, dtype=np.float32)
    a = max(1, int(0.006 * SR)); rl = max(1, int(min(0.09, dur * 0.6) * SR))
    envelope[:a] = np.linspace(0, 1, a)
    envelope[-rl:] *= np.linspace(1, 0, rl)
    return tone * envelope * 0.45

def tokens_to_midi(tokens, bpm=FUNK_BPM, div=480):
    """Note tokens -> .mid for any DAW."""
    import struct as _st
    def vlq(n):
        out = bytearray([n & 0x7F]); n >>= 7
        while n:
            out.insert(0, (n & 0x7F) | 0x80); n >>= 7
        return bytes(out)
    trk, wait = bytearray(), 0
    us = int(60_000_000 / bpm)
    trk += b"\x00\xff\x51\x03" + us.to_bytes(3, "big")      # tempo
    for tok in tokens.split():
        if "/" not in tok:
            continue
        name, _, dur = tok.partition("/")
        try:
            ticks = int(dur) * div // 4
        except ValueError:
            continue
        if name == "R":
            wait += ticks; continue
        try:
            note = name_midi(name)
        except (ValueError, IndexError):
            wait += ticks; continue
        trk += vlq(wait) + bytes([0x90, note, 100])
        trk += vlq(ticks) + bytes([0x80, note, 0])
        wait = 0
    trk += b"\x00\xff\x2f\x00"
    return (b"MThd" + _st.pack(">IHHH", 6, 0, 1, div)
            + b"MTrk" + _st.pack(">I", len(trk)) + bytes(trk))

def synth_funk(tokens, bpm=None, seed=0, arrange=False, stereo=False,
               human=True, genre="funk"):
    """Hook over a tamborzão beat, 808 on the root."""
    r = np.random.default_rng(seed)
    g = GENRES.get(genre, GENRES["funk"])
    spb = 60.0 / (bpm or g["bpm"]) / 4.0                 # seconds per sixteenth
    swing = g["swing"] * spb                             # push the off-beats late
    kick_pat = g["kick"] or KICK_VARIANTS[int(r.integers(0, len(KICK_VARIANTS)))]
    clap_pat = g["clap"] or CLAP_VARIANTS[int(r.integers(0, len(CLAP_VARIANTS)))]

    # lay out the melody and note where each bar starts
    notes, t, bars, bar_first = [], 0.0, [0.0], []
    pending = None
    for tok in tokens.split():
        if tok == "|":
            bars.append(t)
            bar_first.append(pending)
            pending = None
            continue
        if "/" not in tok:
            continue
        name, _, dur = tok.partition("/")
        try:
            length = int(dur) * spb
        except ValueError:
            continue
        if name != "R":
            try:
                midi = name_midi(name)
            except (ValueError, IndexError):
                t += length; continue
            notes.append((t, length, midi))
            if pending is None:
                pending = midi
        t += length
    bar_first.append(pending)

    total = max(t, 4 * spb) + 0.6
    n_total = int(total * SR) + 1
    low_bus = np.zeros(n_total, dtype=np.float32)     # kick + 808
    top_bus = np.zeros(n_total, dtype=np.float32)     # clap, hats, lead

    def place(sig, start, gain=1.0, bus=None):
        target = low_bus if bus == "low" else top_bus
        i = max(0, int(start * SR))          # a nudge can push bar 0 before zero
        end = min(n_total, i + len(sig))
        if end > i:
            target[i:end] += sig[:end - i] * gain

    # drums, bar by bar
    kick = _kick()
    n_bars = max(1, int(np.ceil(total / (16 * spb))))
    # Real drummers push, drag and hit unevenly; a perfect grid sounds machine.
    hum = (lambda: float(r.normal(0, 0.006))) if human else (lambda: 0.0)
    vel = (lambda: float(r.uniform(0.82, 1.12))) if human else (lambda: 1.0)
    for bar in range(n_bars):
        base = bar * 16 * spb
        sw = lambda st: base + st * spb + (swing if st % 2 else 0) + hum()
        for step in kick_pat:
            place(kick, sw(step), 1.35 * vel(), "low")
        for step in clap_pat:
            place(_clap(seed=bar * 7 + step), sw(step), 0.5 * vel())
        if human and bar and bar % 4 == 3:               # a fill before the turn
            for k, step in enumerate((12, 13, 14, 15)):
                place(_clap(seed=bar * 31 + k), base + step * spb,
                      0.28 + 0.09 * k)
        for step in range(0, 16, g["hat"]):              # hats sit well back
            place(_hat(seed=bar * 13 + step, open_=(step == 14)),
                  sw(step), 0.22 * vel())
            if r.random() < 0.18:
                place(_hat(seed=bar * 17 + step), base + (step + 1) * spb, 0.14)

    # 808 follows the first note of each bar
    for bar in range(n_bars):
        midi = bar_first[bar % len(bar_first)] if bar_first else None
        if midi is None:
            continue
        while midi > 52:
            midi -= 12                                   # keep the sub low
        if not (arrange and bar == 0):
            place(_sub(midi, 16 * spb * g["sub"]), bar * 16 * spb, 1.25, "low")

    quiet = set()
    if arrange:
        # bar 0 drums-only, one bar mid-track drops out: stops it sounding looped
        quiet = {0, max(1, n_bars // 2)}
    for i, (start, dur, midi) in enumerate(notes):       # stabs, with space between
        if int(start / (16 * spb)) in quiet:
            continue
        step = int(round(start / spb)) % 16
        if step not in TRESILLO and step % 2 and r.random() < 0.45:
            continue                                     # thin it out
        place(_lead(midi, min(dur, 3 * spb)), start, 0.42)

    # Bisect the top-bus gain to hit the low-end target: six passes.
    def low_share(gain):
        mix = low_bus + top_bus * gain
        spec = np.abs(np.fft.rfft(mix))
        freqs = np.fft.rfftfreq(len(mix), 1 / SR)
        total_e = spec.sum() or 1.0
        return spec[(freqs > 25) & (freqs < 140)].sum() / total_e

    lo_g, hi_g = 0.15, 4.0
    if low_share(1.0) < FUNK_LOW_TARGET:
        lo_g, hi_g = 0.15, 1.0                        # too bright: pull the top down
    else:
        lo_g, hi_g = 1.0, 4.0                         # too dark: bring it up
    for _ in range(6):
        mid = (lo_g + hi_g) / 2
        if low_share(mid) > FUNK_LOW_TARGET:
            lo_g = mid
        else:
            hi_g = mid
    buf = low_bus + top_bus * ((lo_g + hi_g) / 2)

    peak = np.abs(buf).max() or 1.0
    buf = np.tanh(buf / peak * 1.2) * 0.88

    import wave as _wave
    out = io.BytesIO()
    if not stereo:
        pcm = (buf * 32767).astype(np.int16)
        ch = 1
    else:
        # Kick and 808 centred, hats and claps wide, delayed taps per side
        wide = top_bus / (np.abs(top_bus).max() or 1.0) * 0.55
        left = buf.copy(); right = buf.copy()
        for delay, gain, side in ((0.031, .28, 0), (0.047, .22, 1),
                                  (0.071, .15, 0), (0.093, .11, 1)):
            d = int(delay * SR)
            tail = np.zeros_like(wide)
            tail[d:] = wide[:-d] * gain
            (left if side == 0 else right)[:] += tail
        pan = np.stack([left, right], axis=1)
        pan = np.tanh(pan * 1.05) * 0.9
        pcm = (pan * 32767).astype(np.int16)
        ch = 2
    with _wave.open(out, "wb") as w:
        w.setnchannels(ch); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return out.getvalue()

# --- learning from real songs
# Real melodies. MIDI is MThd + MTrk tracks of delta-time and event bytes.
# Point MYCODER_MIDI at .mid you may use.

MIDI_DIR = env("MYCODER_MIDI", "midi")

def _varlen(data, i):
    """MIDI's variable-length integers: 7 bits per byte, top bit means continue."""
    val = 0
    while True:
        b = data[i]; i += 1
        val = (val << 7) | (b & 0x7F)
        if not b & 0x80:
            return val, i

def parse_midi(data):
    """-> (ticks_per_beat, [tracks]); a track is [(tick, midi, on)]."""
    import struct
    if data[:4] != b"MThd":
        raise ValueError("not a MIDI file")
    _, fmt, ntrk, div = struct.unpack(">IHHH", data[4:14])
    if div & 0x8000:
        raise ValueError("SMPTE timing not supported")
    pos, tracks = 14, []
    while pos < len(data) and len(tracks) < ntrk:
        if data[pos:pos + 4] != b"MTrk":
            break
        length = struct.unpack(">I", data[pos + 4:pos + 8])[0]
        body, pos = data[pos + 8:pos + 8 + length], pos + 8 + length
        events, i, tick, status = [], 0, 0, 0
        while i < len(body):
            delta, i = _varlen(body, i)
            tick += delta
            if i >= len(body):
                break
            b = body[i]
            if b & 0x80:
                status = b; i += 1
            if status == 0xFF:                      # meta event
                i += 1
                ln, i = _varlen(body, i)
                i += ln
            elif status in (0xF0, 0xF7):            # sysex
                ln, i = _varlen(body, i)
                i += ln
            else:
                kind = status & 0xF0
                if kind in (0x80, 0x90):
                    note, vel = body[i], body[i + 1]; i += 2
                    events.append((tick, note, kind == 0x90 and vel > 0))
                elif kind in (0xA0, 0xB0, 0xE0):
                    i += 2
                elif kind in (0xC0, 0xD0):
                    i += 1
                else:
                    i += 1
        if events:
            tracks.append(events)
    return div, tracks

def midi_to_tokens(data, grid=4):
    """Melody track -> 'C4/2 R/1 ...' tokens on a sixteenth grid.

    grid=4 means four steps per beat. The melody is taken as the track with the
    most notes, and where notes overlap we keep the highest — that's the tune.
    """
    div, tracks = parse_midi(data)
    if not tracks:
        return ""
    # the melody is the higher line; the other track is chords/bass. Picking by
    # note count alone grabs the accompaniment, which has more notes.
    def score(t):
        pitches = [n for _, n, on in t if on]
        if len(pitches) < 12:
            return -1
        return sum(pitches) / len(pitches)
    track = max(tracks, key=score)

    notes, held = [], {}
    for tick, note, on in track:
        if on:
            held[note] = tick
        elif note in held:
            start = held.pop(note)
            if tick > start:
                notes.append((start, tick, note))
    if not notes:
        return ""
    notes.sort()

    step = div / grid                                 # ticks per grid step
    out, cursor = [], notes[0][0]
    for start, end, note in notes:
        if start < cursor - step / 2:
            continue                                  # overlapping harmony: skip
        gap = round((start - cursor) / step)
        if gap > 0:
            while gap > 0:
                d = min(gap, 8)
                out.append(f"R/{d}")
                gap -= d
        dur = max(1, min(8, round((end - start) / step)))
        out.append(f"{_name(note)}/{dur}")
        cursor = start + dur * step
    return " ".join(out)

_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
def _name(midi):
    return f"{_NAMES[midi % 12]}{midi // 12 - 1}"

struct_error = __import__("struct").error

def build_corpus(folder, limit=None, bars_per_line=8):
    """Every MIDI in a folder -> one big token string, bar-separated."""
    files = []
    for root, _, names in os.walk(folder):
        files += [os.path.join(root, n) for n in sorted(names) if n.lower().endswith(".mid")]
    if limit:
        files = files[:limit]
    lines, kept = [], 0
    for path in files:
        try:
            with open(path, "rb") as f:
                toks = midi_to_tokens(f.read())
        except (ValueError, IndexError, struct_error):
            continue                                  # malformed file, skip it
        if len(toks.split()) < 24:
            continue
        # insert bar lines every 16 sixteenths so the format matches the rest
        parts, count, chunk = [], 0, []
        for t in toks.split():
            chunk.append(t)
            count += int(t.split("/")[1])
            if count >= 16:
                parts.append(" ".join(chunk) + " |")
                chunk, count = [], 0
        if chunk:
            parts.append(" ".join(chunk) + " |")
        lines.append(" ".join(parts))
        kept += 1
    return "\n".join(lines), kept, len(files)

class MusicTrainer:
    """A second NanoGPT trained on note tokens."""

    def __init__(self, block=64, batch=16, lr=2e-3, style=None):
        unpack_embedded()
        self.style = style or MUSIC_STYLE
        if self.style == "real":
            text, kept, seen = build_corpus(MIDI_DIR)
            self.source = f"{kept} MIDI tunes"
            self.fallback = kept == 0
            if self.fallback:                        # no MIDI here: generation only
                text = funk_corpus()
                self.source = f"no MIDI in {MIDI_DIR}/ — generating from saved weights"
        else:
            text = funk_corpus()
            self.source, self.fallback = self.style, False
        self.vocab, self.state_note = None, ""
        if os.path.exists(MUSIC_VOCAB):
            saved = NoteVocab.load(MUSIC_VOCAB)
            cov = 1.0 if self.fallback else saved.coverage(text)
            # A vocabulary from other data drops unknown tokens silently.
            if cov > 0.98:
                self.vocab = saved
            else:
                self.state_note = f"saved vocabulary covered {cov:.0%} of the corpus - rebuilt"
        if self.vocab is None:
            self.vocab = NoteVocab().build(text)
            self.vocab.save(MUSIC_VOCAB)
        ids = np.array([i for i in self.vocab.encode(text)], dtype=np.int64)
        cut = int(0.95 * len(ids))
        self.train_ids, self.val_ids = ids[:cut], ids[cut:]
        self.block, self.batch = block, batch
        self.model = NanoGPT(vocab=len(self.vocab), block=block,
                             n_layer=3, n_head=4, n_embd=96, seed=11)
        self.opt = Adam(self.model.p, lr=lr)
        self.rng = np.random.default_rng()
        self.state = {"step": 0, "loss": None, "val": None, "best_val": None,
                      "params": self.model.n_params(), "vocab": len(self.vocab),
                      "tokens": int(len(ids)), "style": self.style,
                      "source": self.source, "note": self.state_note}
        self.model_missing = not os.path.exists(MUSIC_CKPT)
        if os.path.exists(MUSIC_CKPT):
            try:
                extra = self.model.load(MUSIC_CKPT)
                self.state["step"] = int(extra.get("step", 0))
                self.state["note"] = f"resumed at step {self.state['step']}"
            except Exception as e:
                self.state["note"] = f"music checkpoint unusable ({e})"
                self.model_missing = True

    def val_loss(self, iters=6):
        """Held-out loss; rising while train falls = memorizing."""
        tot = 0.0
        for _ in range(iters):
            ix = self.rng.integers(0, len(self.val_ids) - self.block - 1, self.batch)
            x = np.stack([self.val_ids[i:i + self.block] for i in ix])
            y = np.stack([self.val_ids[i + 1:i + 1 + self.block] for i in ix])
            tot += self.model.forward(x, y)[1]
        return tot / iters

    def step_once(self):
        ix = self.rng.integers(0, len(self.train_ids) - self.block - 1, self.batch)
        x = np.stack([self.train_ids[i:i + self.block] for i in ix])
        y = np.stack([self.train_ids[i + 1:i + 1 + self.block] for i in ix])
        _, loss, cache = self.model.forward(x, y)
        self.opt.step(self.model.p, self.model.backward(cache))
        self.state["step"] += 1
        self.state["loss"] = round(loss, 4)
        if self.state["step"] % 100 == 0:
            vl = round(self.val_loss(), 4)
            self.state["val"] = vl
            if self.state["best_val"] is None or vl < self.state["best_val"]:
                self.state["best_val"] = vl
        return loss

    def save(self):
        tmp = MUSIC_CKPT + ".tmp"
        self.model.save(tmp, extra={"step": self.state["step"]})
        os.replace(tmp + ".npz", MUSIC_CKPT)

    BEST_TEMP = {"funk": 0.75, "real": 0.9}

    # Over 8 hooks: generator 98% in key / 54% repeat, model 95% / 42%. The
    # generator wins on funk (the model only imitates it); the model earns
    # its keep on 'real'.
    BEST_ENGINE = {"funk": "generator", "real": "model"}

    def compose(self, bars=8, temp=None, seed=None, bpm=None, wave="soft",
                engine=None, arrange=False, stereo=False, genre="funk"):
        """Temps are measured: funk 0.75 gives 94% in key, 4% copying."""
        # measured per model: funk copies its corpus at 0.65 and drifts out of key
        # at 0.9; the folk model needs 0.9 to keep its stepwise motion.
        temp = self.BEST_TEMP.get(self.style, 0.8) if temp is None else temp
        engine = engine or self.BEST_ENGINE.get(self.style, "model")
        if self.model_missing and engine == "model":
            engine = "generator"                 # no weights here: use the rule
        rng = np.random.default_rng(seed)
        if engine == "generator":
            # one hook, looped — stitching different hooks together changes key
            # mid-track and drops in-key from 98% to 84%. Funk loops one idea.
            hook = make_funk_hook(int(rng.integers(0, 10 ** 6)), genre).split()
            need = bars * 16
            toks = list(hook)
            while len(toks) < need:
                toks += hook
            tokens = " ".join(toks[:need])
            return synth_funk(tokens, bpm=bpm, arrange=arrange, stereo=stereo,
                              genre=genre,
                              seed=int(rng.integers(0, 10_000))), tokens
        funk = self.style in ("funk", "real")   # both play over the tamborzão beat
        # seed with the same kind of music the model was trained on — priming the
        # folk model with a funk hook sends it somewhere it has never been
        seed_text = (make_funk_hook if self.style == "funk" else make_melody)(
            int(rng.integers(0, 10_000)))
        start = self.vocab.encode(" ".join(seed_text.split()[:8])) or [1]
        ids = self.model.generate(start, max_new_tokens=bars * 16,
                                  temperature=temp, top_k=12, rng=rng)
        tokens = " ".join(self.vocab.decode(ids).split()[:bars * 16])
        if funk:
            return synth_funk(tokens, bpm=bpm, arrange=arrange, stereo=stereo,
                              genre=genre,
                              seed=int(rng.integers(0, 10_000))), tokens
        return synth_funk(tokens, bpm=bpm, genre=genre), tokens

# --- the internet
# Answer "search this", and pull pages into the corpus — data is what the
# model is short of.

UA = "Mozilla/5.0 (compatible; mycoder/1.0)"

def strip_html(html):
    """HTML -> text; drops script/style, unescapes entities."""
    import html as _html
    import re
    html = re.sub(r"(?is)<(script|style|nav|footer|header)[^>]*>.*?</\1>", " ", html)
    html = re.sub(r"(?is)<br\s*/?>|</p>|</div>|</li>|</h[1-6]>", "\n", html)
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    text = _html.unescape(text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return re.sub(r"\n\s*\n\s*\n+", "\n\n", text).strip()

def fetch_page(url, timeout=20, limit=200_000):
    """Download a page and return readable text."""
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read(limit)
        charset = r.headers.get_content_charset() or "utf-8"
    body = raw.decode(charset, errors="replace")
    return strip_html(body) if "<" in body[:2000] else body

GOOGLE_KEY = os.environ.get("GOOGLE_API_KEY")
GOOGLE_CX  = os.environ.get("GOOGLE_CX")

def google_search(query, count=5, timeout=20):
    """Google Custom Search JSON API; needs GOOGLE_API_KEY + GOOGLE_CX (free
    tier 100/day). Scraping google.com is blocked and against their terms."""
    from urllib.parse import quote_plus
    if not (GOOGLE_KEY and GOOGLE_CX):
        raise RuntimeError("no GOOGLE_API_KEY / GOOGLE_CX set")
    url = (f"https://www.googleapis.com/customsearch/v1?key={GOOGLE_KEY}"
           f"&cx={GOOGLE_CX}&num={min(count, 10)}&q={quote_plus(query)}")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        payload = json.loads(r.read().decode())
    return [{"title": it.get("title", ""), "url": it.get("link", ""),
             "snippet": it.get("snippet", "")}
            for it in payload.get("items", [])][:count]

def web_search(query, count=5, timeout=20):
    """Google when a key is configured, DuckDuckGo's HTML endpoint otherwise."""
    if GOOGLE_KEY and GOOGLE_CX:
        try:
            hits = google_search(query, count, timeout)
            if hits:
                return hits
        except Exception:
            pass                                  # fall through rather than fail

    import re
    from urllib.parse import quote_plus, unquote, urlparse, parse_qs
    url = "https://html.duckduckgo.com/html/?q=" + quote_plus(query)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        page = r.read(400_000).decode("utf-8", errors="replace")

    out = []
    for m_ in re.finditer(r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
                          page, re.S):
        href, title = m_.group(1), strip_html(m_.group(2))
        if "uddg=" in href:                              # unwrap the redirect
            qs = parse_qs(urlparse(href).query)
            href = unquote(qs.get("uddg", [href])[0])
        if href.startswith("http"):
            out.append({"title": title, "url": href, "snippet": ""})
        if len(out) >= count:
            break

    for i, sn in enumerate(re.findall(r'(?is)<a[^>]+result__snippet[^>]*>(.*?)</a>', page)):
        if i < len(out):
            out[i]["snippet"] = strip_html(sn)[:300]
    return out

# --- tests

def run_tests():
    """Finite-difference check on every parameter, then a memorization run."""
    rng = np.random.default_rng(0)
    m = NanoGPT(vocab=11, block=6, n_layer=2, n_head=2, n_embd=8, seed=1)
    for k in m.p:
        m.p[k] = m.p[k].astype(np.float64)              # clean numeric derivative
    idx, tgt = rng.integers(0, 11, (2, 6)), rng.integers(0, 11, (2, 6))
    grads = m.backward(m.forward(idx, tgt)[2])

    eps, worst, where = 1e-5, 0.0, None
    for key in m.p:
        flat = m.p[key].reshape(-1)
        for i in rng.choice(flat.size, size=min(6, flat.size), replace=False):
            orig = flat[i]
            flat[i] = orig + eps; lp = m.forward(idx, tgt)[1]
            flat[i] = orig - eps; lm = m.forward(idx, tgt)[1]
            flat[i] = orig
            numeric, analytic = (lp - lm) / (2 * eps), grads[key].reshape(-1)[i]
            rel = abs(numeric - analytic) / max(abs(numeric), abs(analytic), 1e-8)
            if rel > worst:
                worst, where = rel, f"{key}[{i}]"
    grad_ok = worst < 2e-3
    print(f"gradient check: worst relative error {worst:.2e} at {where}"
          f" -> {'PASS' if grad_ok else 'FAIL'}")

    text = "def evaluate(board):\n    return sum(piece_value(p) for p in board)\n"
    stoi = {c: i for i, c in enumerate(sorted(set(text)))}
    ids = np.array([stoi[c] for c in text], dtype=np.int64)
    blk = 16
    m2 = NanoGPT(vocab=len(stoi), block=blk, n_layer=2, n_head=2, n_embd=32, seed=2)
    opt = Adam(m2.p, lr=3e-3)
    x = np.stack([ids[i:i + blk] for i in range(len(ids) - blk - 1)])
    y = np.stack([ids[i + 1:i + 1 + blk] for i in range(len(ids) - blk - 1)])
    first = last = None
    for s in range(300):
        _, loss, cache = m2.forward(x, y)
        opt.step(m2.p, m2.backward(cache))
        first = loss if s == 0 else first
        last = loss
    fit_ok = last < 0.2 and last < first / 4
    print(f"overfit check: loss {first:.3f} -> {last:.3f} -> {'PASS' if fit_ok else 'FAIL'}")

    # 7. note names must survive a round trip, or every melody is transposed junk
    note_ok = all(name_midi(midi_name(n)) == n for n in range(24, 108))
    print(f"note name round trip: {'PASS' if note_ok else 'FAIL'}")

    # 8. training melodies should sit inside a single scale
    mel = make_melody(1)
    pitches = [name_midi(t.split("/")[0]) % 12 for t in mel.split()
               if "/" in t and not t.startswith("R")]
    fit = max(max(sum(1 for p in pitches if (p - r) % 12 in sc) for r in range(12))
              for sc in (MAJOR, MINOR))
    key_ok = fit == len(pitches)
    print(f"melodies stay in key: {fit}/{len(pitches)} -> {'PASS' if key_ok else 'FAIL'}")

    # 9. the synthesizer must emit a valid, audible wav
    import wave as _w
    wav = synth_funk(mel)
    with _w.open(io.BytesIO(wav)) as f:
        pcm = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
        audio_ok = f.getframerate() == SR and f.getnframes() > SR and np.abs(pcm).max() > 10000
    print(f"synthesizer output: {len(wav)/1024:.0f} KB, {'PASS' if audio_ok else 'FAIL'}")

    # 11. the funk mix should be bass-forward
    import wave as _wv
    with _wv.open(io.BytesIO(synth_funk(make_funk_hook(2), seed=1))) as fh:
        sig = np.frombuffer(fh.readframes(fh.getnframes()), dtype=np.int16).astype(np.float32)
    spec = np.abs(np.fft.rfft(sig))
    frq = np.fft.rfftfreq(len(sig), 1 / SR)
    low_share = spec[(frq > 25) & (frq < 140)].sum() / spec.sum()
    # the renderer balances to FUNK_LOW_TARGET, so this checks it actually lands
    bass_ok = abs(low_share - FUNK_LOW_TARGET) < 0.03
    print(f"funk low-end share: {low_share:.0%} "
          f"(target {FUNK_LOW_TARGET:.0%}) -> {'PASS' if bass_ok else 'FAIL'}")

    # 12. every snippet must actually run — the point of the rewrite
    ran = 0
    for keys, code in SNIPPETS:
        try:
            exec(compile(code, "<snippet>", "exec"), {"__name__": "snippet"})
            ran += 1
        except Exception as exc:
            print(f"  snippet {keys[0]!r} failed: {exc}")
    skel = write_code("a function called tidy")[0]
    try:
        exec(compile(skel, "<skeleton>", "exec"), {})
        skel_ok = True
    except Exception:
        skel_ok = False
    code_ok = ran == len(SNIPPETS) and skel_ok
    print(f"code snippets run: {ran}/{len(SNIPPETS)} + skeleton -> "
          f"{'PASS' if code_ok else 'FAIL'}")

    # 13. each genre must render its own kick pattern, not funk relabelled
    import wave as _w2
    saved_c, saved_h = _clap, _hat
    globals()["_clap"] = globals()["_hat"] = lambda *a, **k: np.zeros(64, np.float32)
    try:
        rest = " ".join(["R/8 R/8 |"] * 4)
        genre_ok = True
        for gname, gspec in GENRES.items():
            if not gspec["kick"]:
                continue
            with _w2.open(io.BytesIO(synth_funk(rest, seed=2, human=False,
                                                genre=gname))) as fh:
                amp = np.abs(np.frombuffer(fh.readframes(fh.getnframes()),
                                           dtype=np.int16).astype(np.float32))
            sp = 60 / gspec["bpm"] / 4
            got = []
            for st in range(16):
                k = int(st * sp * SR)
                here, prev = amp[k:k + 300], amp[max(0, k - 400):k]
                if len(here) and here.max() > 8000 and (not len(prev)
                                                        or prev.max() < 8000):
                    got.append(st)
            genre_ok &= set(got) == set(gspec["kick"])
    finally:
        globals()["_clap"], globals()["_hat"] = saved_c, saved_h
    print(f"genre kick patterns: {'PASS' if genre_ok else 'FAIL'}")

    # 14. natural phrasing, not just the words I happened to pick
    natural = [("gimme a beat", "music"), ("drop a house track", "music"),
               ("can you flip a word backwards", "code"), ("whats the average", "code"),
               ("what does that mean", "explain"), ("hows it going", "greet"),
               ("cheers mate", "thanks"), ("what can u do", "help"),
               ("how were you made", "status"), ("i want the midi", "midi")]
    miss = [(m, detect_intent(m), w) for m, w in natural if detect_intent(m) != w]
    natural_ok = not miss
    print(f"natural phrasing: {len(natural)-len(miss)}/{len(natural)} -> "
          f"{'PASS' if natural_ok else 'FAIL ' + str(miss)}")

    # 15. the paid tier must gate, and admins must get in free
    _t = Trainer(verbose=False)
    blocked = milo_reply("write a fizzbuzz", _t, "nova-iris",
                         ChatSession())["intent"]
    admin = milo_reply("write a fizzbuzz", _t, "nova-iris",
                       ChatSession("editornova"))
    tier_ok = blocked == "locked" and "code" in admin and "checked" in admin
    print(f"paid gate, admin free, deep check: {'PASS' if tier_ok else 'FAIL'}")

    # 16. the chat router must map plain phrasing to the right capability
    cases = [("make me a funk beat", "music"), ("search for baile funk", "search"),
             ("make it faster", "music"),   # a follow-up, not a code ask
             ("hey", "greet"), ("what can you do", "help"),
             ("who are you", "identity"),
             ("finish this code: def f(", "code"), ("qwerty zxcv", "unknown")]
    wrong = [(t, detect_intent(t), want) for t, want in cases if detect_intent(t) != want]
    router_ok = not wrong
    print(f"chat router: {len(cases)-len(wrong)}/{len(cases)} "
          f"-> {'PASS' if router_ok else 'FAIL ' + str(wrong)}")

    # 13. html stripping
    html_ok = strip_html("<p>hi <b>there</b></p><script>bad()</script>") == "hi there"
    print(f"html to text: {'PASS' if html_ok else 'FAIL'}")

    ok = (grad_ok and fit_ok and note_ok and key_ok and audio_ok and html_ok
          and bass_ok and router_ok and code_ok and genre_ok and natural_ok
          and tier_ok)
    print("\nall good" if ok else "\nsomething is wrong — do not train with this")
    return ok

# --- cli

def cli():
    ap = argparse.ArgumentParser(description="a coding AI in one file")
    ap.add_argument("command", nargs="?", default="serve",
                    choices=["serve", "test", "chat",
                             "music", "train-music", "search"])
    ap.add_argument("prompt", nargs="*", help="text for search / learn")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--temp", type=float, default=None)
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 5001)))
    ap.add_argument("--out", default="out", help="output file")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--bars", type=int, default=8)
    ap.add_argument("--bpm", type=int, default=None)
    ap.add_argument("--style", default=None, help="funk (default), real, or melodic")
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS),
                    help="milo-1.1 or milo-1.1-pro")
    ap.add_argument("--engine", default=None, choices=["model", "generator"],
                    help="generator needs no weights; scores better on funk")
    ap.add_argument("--pages", type=int, default=3)
    args = ap.parse_args()

    if args.command == "test":
        raise SystemExit(0 if run_tests() else 1)

    if args.command == "search":
        q = " ".join(args.prompt)
        if not q:
            raise SystemExit('Give it something: nova.py search "python chess"')
        for r in web_search(q, count=5):
            print(f"\n{r['title']}\n  {r['url']}\n  {r['snippet'][:160]}")
        return

    if args.command in ("music", "train-music"):
        mt = MusicTrainer(style=args.style)
        print(f"{mt.state['style']} | {mt.state['params']:,} params | vocab {mt.state['vocab']} | "
              f"{mt.state['note'] or 'fresh start'}")
        if args.command == "train-music":
            t0 = time.time()
            for s_ in range(args.steps):
                loss = mt.step_once()
                if s_ % 250 == 0 or s_ == args.steps - 1:
                    print(f"step {mt.state['step']:6d} | loss {loss:.4f}"
                          f" | {(time.time()-t0)/60:.1f}m")
            mt.save()
            print(f"saved to {MUSIC_CKPT}")
        else:
            path = args.out if args.out.endswith(".wav") else args.out + ".wav"
            wav, tokens = mt.compose(bars=args.bars, temp=args.temp, seed=args.seed,
                                     bpm=args.bpm, engine=args.engine)
            with open(path, "wb") as f:
                f.write(wav)
            print(tokens[:200] + ("…" if len(tokens) > 200 else ""))
            print("wrote", path)
        return

    t = Trainer(verbose=True)
    if t.music is None:
        raise SystemExit(t.state.get("note") or "no music model")
    print(t.state.get("note") or "ready")

    if args.command == "chat":
        spec = MODELS[args.model]
        ses = get_session("cli")
        print(f"{spec['name']} — {spec['blurb']}\nType help, or Ctrl-C to leave.\n")
        while True:
            try:
                msg = input("you > ")
            except (EOFError, KeyboardInterrupt):
                print(); return
            r = milo_reply(msg, t, args.model, ses)
            print(f"{spec['name'].lower().replace(' ', '')} > {r['reply']}")
            if "code" in r:
                print("\n" + r["code"] + "\n")
            if r.get("links"):
                for l in r["links"][:5]:
                    print(f"  {l['title'][:60]}\n    {l['url'][:70]}")
            if "audio" in r:
                import base64
                path = f"milo-{int(time.time())}.wav"
                with open(path, "wb") as f:
                    f.write(base64.b64decode(r["audio"].split(",")[1]))
                print(f"  saved {path}")
            print()

    else:
        if AUTOTRAIN:
            t.start()
        print(f"http://127.0.0.1:{args.port}")
        build_app(t).run(host="0.0.0.0", port=args.port, debug=False, use_reloader=False)

# --- trained weights (separate files)
# Weights ship separately (2.7 MB) in MYCODER_DATA_DIR, or MYCODER_CKPT_URL.

EMBEDDED = {}

# gunicorn entry point: `gunicorn nova:app --workers 1 --threads 2`
if os.environ.get("SERVER_SOFTWARE", "").startswith("gunicorn"):
    trainer = Trainer()
    if AUTOTRAIN:
        trainer.start()
    app = build_app(trainer)

if __name__ == "__main__":
    cli()
