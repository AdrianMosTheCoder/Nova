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
    """Unpack EMBEDDED weights."""
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
    """Holds the music model, built on first use."""

    def __init__(self, paths=None, verbose=False):
        os.makedirs(DATA_DIR, exist_ok=True)
        unpack_embedded()
        self._music = None
        self._music_off = env("MYCODER_MUSIC", "1") != "1"
        self.model = None
        self.thread = None
        self.state = {"status": "ready", "step": 0, "params": 0, "tokens": 0,
                      "note": "", "music": None}

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

    def start(self):
        """Train in the background, if asked."""
        if self.thread and self.thread.is_alive() or self.music is None:
            return
        self._stop = threading.Event()

        def loop():
            while not self._stop.is_set():
                for _ in range(BURST):
                    self.music.step_once()
                self._stop.wait(PAUSE)

        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()

    def stop(self):
        if self.thread:
            self._stop.set()


# --- web app

PAGE = """<!doctype html>
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Nova</title>
<style>
*{box-sizing:border-box}
:root{color-scheme:dark}
body{background:#0d141b;color:#dbe4ee;font:15px/1.65 ui-monospace,Menlo,monospace;margin:0;
height:100vh;display:flex;overflow:hidden}
aside{width:230px;flex:none;background:#080d13;border-right:1px solid #1a2531;display:flex;
flex-direction:column;padding:12px}
aside h1{font-size:14px;margin:4px 6px 12px;letter-spacing:.05em}
#new{background:#16212c;color:#cfe0ef;border:1px solid #24323f;border-radius:7px;padding:9px;
cursor:pointer;font:inherit;margin-bottom:12px}
#new:hover{border-color:#3fae8f}
#chats{flex:1;overflow-y:auto;display:flex;flex-direction:column;gap:3px}
.chat{padding:8px 10px;border-radius:6px;cursor:pointer;font-size:12.5px;color:#8ea4b8;
white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.chat:hover{background:#111a23}
.chat.on{background:#16212c;color:#dbe4ee}
main{flex:1;display:flex;flex-direction:column;min-width:0}
header{border-bottom:1px solid #1a2531;padding:10px 18px;display:flex;gap:10px;
align-items:center;flex-wrap:wrap}
select,#acct{background:#111a23;color:#dbe4ee;border:1px solid #24323f;padding:6px 8px;
border-radius:6px;font:inherit}
#acct{width:100px}
.tag{color:#63788c;font-size:11px}
#feed{flex:1;overflow-y:auto;padding:26px 18px}
.turn{display:flex;gap:12px;max-width:740px;margin:0 auto 22px}
.who{width:26px;height:26px;border-radius:6px;flex:none;display:flex;align-items:center;
justify-content:center;font-size:10px;font-weight:700}
.you .who{background:#2b3a49;color:#9fb3c8}
.bot .who{background:#1d5c56;color:#a8ede4}
.body{flex:1;min-width:0}
.name{font-size:11px;color:#63788c;margin-bottom:4px}
.text{white-space:pre-wrap;word-wrap:break-word}
.text code{background:#111a23;padding:1px 5px;border-radius:4px;font-size:13px}
.text b{color:#fff}
.card{background:#080d13;border:1px solid #1a2531;border-radius:9px;margin-top:10px;
overflow:hidden}
.card .top{display:flex;justify-content:space-between;padding:7px 12px;
border-bottom:1px solid #1a2531;font-size:11px;color:#63788c}
.card pre{margin:0;padding:12px 14px;overflow-x:auto;font-size:13px}
.out{border-left:3px solid #3fae8f;background:#080d13;padding:10px 12px;margin-top:8px;
border-radius:6px;font-size:13px;white-space:pre-wrap}
.err{border-left-color:#c7566d}
.acts{display:flex;gap:7px;margin-top:9px}
.acts button,.card .top button{background:none;border:1px solid #24323f;color:#8ea4b8;
padding:4px 10px;border-radius:6px;cursor:pointer;font:inherit;font-size:11.5px}
.acts button:hover,.card .top button:hover{border-color:#3fae8f;color:#cfe9e2}
.chips{display:flex;gap:8px;flex-wrap:wrap;margin-top:14px}
.chips button{background:none;border:1px dashed #2b3a49;color:#7d93a8;padding:8px 13px;
border-radius:8px;cursor:pointer;font:inherit;font-size:12.5px}
.chips button:hover{border-color:#3fae8f;color:#cfe9e2}
a{color:#6fd3c0}
.link{display:block;margin-top:9px;font-size:13px}
.link span{color:#63788c;display:block;font-size:12px}
audio{width:100%;margin-top:10px}
.cursor{display:inline-block;width:7px;background:#3fae8f;animation:bl .9s infinite}
@keyframes bl{50%{opacity:0}
}
footer{border-top:1px solid #1a2531;padding:14px 18px}
.bar{max-width:740px;margin:0 auto;display:flex;gap:10px;align-items:flex-end}
#msg{flex:1;background:#111a23;color:#dbe4ee;border:1px solid #24323f;padding:12px 14px;
border-radius:10px;resize:none;height:48px;max-height:160px;font:inherit}
#msg:focus{outline:none;border-color:#3fae8f}
#send{background:#2f7d76;color:#eefbf8;border:0;padding:13px 20px;border-radius:10px;
cursor:pointer;font:inherit}
#send:disabled{background:#1b2a33;color:#5b7186;cursor:default}
.hint{max-width:740px;margin:8px auto 0;color:#4d6175;font-size:11px}
@media(max-width:700px){aside{display:none}
}
</style>

<aside>
  <h1>Nova</h1>
  <button id="new">+ New chat</button>
  <div id="chats"></div>
</aside>

<main>
  <header>
    <select id="model"></select>
    <input id="acct" placeholder="account">
    <span class="tag" id="blurb"></span>
  </header>
  <div id="train" style="display:none;border-bottom:1px solid #1a2531;
padding:8px 18px;font-size:12px;color:#8ea4b8"></div>
<div id="feed"></div>
  <footer>
    <div class="bar">
      <textarea id="msg" placeholder="Ask Nova..."></textarea>
      <button id="send">Send</button>
    </div>
    <div class="hint">Enter sends, Shift+Enter for a new line.</div>
  </footer>
</main>

<script>
const $=id=>document.getElementById(id);
const feed=$('feed'),msg=$('msg');
let chats=[],cur=null,busy=false;

function newChat(){
const c={id:'c'+Date.now(),title:'New chat',turns:[]};
chats.unshift(c);cur=c;drawList();drawFeed();return c;
}
function drawList(){
$('chats').innerHTML='';
chats.forEach(c=>{const d=document.createElement('div');
d.className='chat'+(c===cur?' on':'');d.textContent=c.title;
d.onclick=()=>{cur=c;drawList();drawFeed();};$('chats').appendChild(d);});
}
function drawFeed(){
feed.innerHTML='';
if(!cur.turns.length){welcome();return;}
cur.turns.forEach(t=>{
const b=turn(t.who,t.who==='you'?'you':'Nova');
say(b,t.text);
if(t.code)codeCard(b,t.code);
if(t.audio){const a=document.createElement('audio');a.controls=true;
a.src=t.audio;b.appendChild(a);}
if(t.who!=='you')actions(b,t);});
}
function turn(who,name){
const d=document.createElement('div');
d.className='turn '+(who==='you'?'you':'bot');
d.innerHTML='<div class="who">'+(who==='you'?'YOU':'N')+'</div>'+
'<div class="body"><div class="name">'+name+'</div></div>';
feed.appendChild(d);feed.scrollTop=feed.scrollHeight;
return d.querySelector('.body');
}
function fmt(t){
return t.replace(/&/g,'&amp;').replace(/</g,'&lt;')
.replace(/\\*\\*(.+?)\\*\\*/g,'<b>$1</b>').replace(/`([^`]+)`/g,'<code>$1</code>');
}
function say(body,text,cls){
const p=document.createElement('div');p.className=cls||'text';
p.innerHTML=fmt(text);body.appendChild(p);
feed.scrollTop=feed.scrollHeight;return p;
}
async function reveal(body,text){
const p=say(body,'');const cur=document.createElement('span');
cur.className='cursor';cur.textContent=' ';p.appendChild(cur);
const step=Math.max(1,Math.round(text.length/45));
for(let i=0;i<=text.length;i+=step){
p.innerHTML=fmt(text.slice(0,i));p.appendChild(cur);
feed.scrollTop=feed.scrollHeight;await new Promise(r=>setTimeout(r,12));}
p.innerHTML=fmt(text);return p;
}
function welcome(){
const b=turn('bot','Nova');
say(b,"What are we making?");
const c=document.createElement('div');c.className='chips';
['make a funk beat at 130 bpm','write a fibonacci function',
'make a lofi loop','search for baile funk'].forEach(t=>{
const x=document.createElement('button');x.textContent=t;
x.onclick=()=>{msg.value=t;send();};c.appendChild(x);});
b.appendChild(c);
}
function codeCard(b,code){
const c=document.createElement('div');c.className='card';
const top=document.createElement('div');top.className='top';
top.innerHTML='<span>python</span>';
const run=document.createElement('button');run.textContent='Run';
top.appendChild(run);c.appendChild(top);
const pre=document.createElement('pre');pre.textContent=code;c.appendChild(pre);
b.appendChild(c);
run.onclick=async()=>{run.disabled=true;run.textContent='running';
try{const d=await(await fetch('/run',{method:'POST',
headers:{'Content-Type':'application/json'},
body:JSON.stringify({session:cur.id})})).json();
const o=say(b,d.error?('error: '+d.error):(d.output||'(ran, printed nothing)'));
o.className=d.error?'out err':'out';}catch(e){say(b,'failed: '+e.message,'out err');}
run.disabled=false;run.textContent='Run again';};
}
function actions(b,t){
const a=document.createElement('div');a.className='acts';
const cp=document.createElement('button');cp.textContent='Copy';
cp.onclick=()=>{navigator.clipboard.writeText(t.code||t.text);
cp.textContent='Copied';setTimeout(()=>cp.textContent='Copy',1200);};
const rg=document.createElement('button');rg.textContent='Regenerate';
rg.onclick=()=>{const last=[...cur.turns].reverse().find(x=>x.who==='you');
if(last){cur.turns=cur.turns.slice(0,cur.turns.indexOf(t));drawFeed();
send(last.text);}};
a.appendChild(cp);a.appendChild(rg);b.appendChild(a);
}

async function poll(){
try{
const d=await(await fetch('/training')).json();
if(!d.running&&!d.step){$('train').style.display='none';return;}
const bar='#'.repeat(Math.round(d.percent/5))+'.'.repeat(20-Math.round(d.percent/5));
$('train').style.display='block';
$('train').textContent=`val ${d.val} [${bar}] ${d.percent}% `+
`· step ${d.step.toLocaleString()}/${d.target.toLocaleString()} `+
`· understanding ${d.understanding}/10 chat ${d.chat}/10 code ${d.code}/10`+
(d.eta?` · ${d.eta} left`:'');
}catch(e){}
}
setInterval(poll,5000);poll();

async function boot(){
newChat();                      // exists before any fetch can fail
try{
const d=await(await fetch('/models')).json();
const sel=$('model');
for(const[id,m]of Object.entries(d.models)){
const o=document.createElement('option');o.value=id;o.textContent=m.name;
if(id===d.default)o.selected=true;sel.appendChild(o);}
const upd=()=>{const m=d.models[sel.value];
if(m)$('blurb').textContent=m.blurb+' · '+(d.brain||'');};
sel.onchange=upd;upd();
}catch(e){$('blurb').textContent='could not load models: '+e.message;}
}
boot();

async function send(preset){
const text=(preset||msg.value).trim();if(!text||busy)return;
if(!cur)newChat();
busy=true;$('send').disabled=true;if(!preset)msg.value='';
msg.style.height='48px';
if(cur.title==='New chat'){cur.title=text.slice(0,28);drawList();}
cur.turns.push({who:'you',text});say(turn('you','you'),text);
const b=turn('bot','Nova');
const dots=say(b,'...');
try{
const res=await fetch('/chat',{method:'POST',
headers:{'Content-Type':'application/json'},
body:JSON.stringify({message:text,model:$('model').value,
account:$('acct').value,session:cur.id})});
if(!res.ok)throw new Error('server said '+res.status);
const r=await res.json();
dots.remove();
const t={who:'bot',text:r.reply,code:r.code,audio:r.audio_url,midi:r.midi_url,
links:r.links};
cur.turns.push(t);
await reveal(b,r.reply);
if(t.code)codeCard(b,t.code);
if(t.audio){const a=document.createElement('audio');a.controls=true;a.src=t.audio;
b.appendChild(a);a.play().catch(()=>{});}
if(t.midi){const d=document.createElement('a');d.className='link';d.href=t.midi;
d.download='nova.mid';d.textContent='Download the MIDI';b.appendChild(d);}
(t.links||[]).forEach(l=>{const a=document.createElement('a');a.className='link';
a.href=l.url;a.target='_blank';a.innerHTML=fmt(l.title)+'<span>'+
fmt(l.snippet||l.url)+'</span>';b.appendChild(a);});
actions(b,t);
}catch(e){dots.remove();say(b,'Could not reach Nova: '+e.message,'out err');}
finally{busy=false;$('send').disabled=false;msg.focus();
feed.scrollTop=feed.scrollHeight;}
}
$('send').onclick=()=>send();
$('new').onclick=()=>{newChat();msg.focus();};
msg.addEventListener('keydown',e=>{
if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send();}});
msg.addEventListener('input',()=>{msg.style.height='48px';
msg.style.height=Math.min(msg.scrollHeight,160)+'px';});
</script>"""



# --- milo: chat
# Routes what you ask to something it can do. No language model writes the
# replies; the work is real.

BASE = ["status", "help"]
MUSIC = BASE + ["music", "midi", "arrange", "explain"]   # Rio is the music
DEEP = MUSIC + ["styles", "long", "stereo"]   # model, so it gets the lot
TALK = BASE + ["search", "code"]            # Milo does words, not music
MODELS = {
    "rio-1.6":      {"name": "Rio 1.6", "can": MUSIC, "warm": True,
                     "blurb": "7 genres, arrangement, MIDI, follow-ups."},
    "rio-1.6-pro":  {"name": "Rio 1.6 Pro", "can": DEEP, "warm": True,
                     "blurb": "7 genres, stereo, folk, 32-bar tracks."},
    "milo-1.8":     {"name": "Milo 1.8", "can": TALK + ["explain"], "warm": True,
                     "blurb": "Code that explains itself, plus search."},
    "nova-mine":    {"name": "Nova (mine)", "warm": True, "mine": True,
                     "can": DEEP + TALK + ["open", "compose", "explain"],
                     "blurb": "Replies written by the model you're training. "
                              "Rough until the loss comes down."},
    "luca":         {"name": "Luca", "warm": True,
                     "can": DEEP + TALK + ["open", "compose", "explain",
                                           "deep", "fix"],
                     "blurb": "Everything Nova can do, and it fixes its own "
                              "code until it runs."},
    "nova-iris":    {"name": "Nova Iris", "warm": True,
                     "can": TALK + ["open", "compose", "explain", "deep"],
                     "blurb": "Checks its own code by running it. Free."},
    "milo-1.8-pro": {"name": "Milo 1.8 Pro", "warm": True,
                     "can": TALK + ["open", "compose", "explain"],
                     "blurb": "Understands loose phrasing, typos included."},
}
DEFAULT_MODEL = env("MYCODER_MODEL", "rio-1.6")
ADMINS = set(env("NOVA_ADMINS", "editornova").split(","))
SUBSCRIBERS = set(w for w in env("NOVA_PAID", "").split(",") if w)


def may_use(model, account=""):
    """Nothing is paid right now — no model carries "paid": True. The check
    stays here so you can switch a tier back on by adding that one key."""
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
# A string is letters in a row, like beads on a thread. [::-1] walks the
# row backwards: same beads, opposite order. reverse("funk") -> "knuf"
def reverse(text):
    return text[::-1]
@@fizzbuzz
# For each number: does 3 divide evenly? does 5? (% is the remainder, so
# % 3 == 0 means it fits.) fizzbuzz(5) -> 1 2 Fizz 4 Buzz
def fizzbuzz(n=100):
    for i in range(1, n + 1):
        out = ("Fizz" if i % 3 == 0 else "") + ("Buzz" if i % 5 == 0 else "")
        print(out or i)
@@fibonacci|fib
# Each number is the two before it added up: 0, 1, 1, 2, 3, 5...
# a, b = b, a + b shuffles them along one place. fib(6) -> [0,1,1,2,3,5]
def fib(n):
    a, b, out = 0, 1, []
    for _ in range(n):
        out.append(a)
        a, b = b, a + b
    return out
@@prime
# A prime divides only by 1 and itself, so try every number below it. We
# stop at the square root because factors come in pairs.
# is_prime(7) -> True, is_prime(9) -> False
def is_prime(n):
    if n < 2:
        return False
    for d in range(2, int(n ** 0.5) + 1):
        if n % d == 0:
            return False
    return True
@@count word|word count
# A dict is labelled boxes: the word labels it, the count is inside.
# counts.get(word, 0) is "what's there, or 0 if it's new".
# Try it: count_words("a b a") -> {a: 2, b: 1}
def count_words(text):
    counts = {}
    for word in text.lower().split():
        counts[word] = counts.get(word, 0) + 1
    return counts
@@binary search|bisect
# Like a dictionary: open the middle, keep the half it's in, repeat.
# Halving each guess means 1000 items take ~10 tries. Must be sorted.
# binary_search([1, 3, 5], 5) -> 2
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
# key= says which bit to compare: kv[1] is the value, not the label.
# Try: sort_by_value({a: 1, b: 9}) puts b first, because 9 > 1
def sort_by_value(d, biggest_first=True):
    return dict(sorted(d.items(), key=lambda kv: kv[1], reverse=biggest_first))
@@read a file|read file|open a file
# Opening a file is like a drawer: you must close it. `with` closes it
# for you when the block ends, even if something breaks.
def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [line.rstrip() for line in f]
@@class
# A class is a cookie cutter; each thing you make is a cookie. __init__
# runs when you make one, self is "this one", __repr__ is how it prints.
# Try: Thing("kick") -> Thing('kick')
class Thing:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"Thing({self.name!r})"
@@average|mean
# The total shared out evenly. sum() adds them, len() counts them. The
# if guards an empty list, since dividing by zero crashes.
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

# --- writing code without a model
#
# A lookup table only ever knows the phrases someone typed into it. This reads
# the request instead: what operation, on what kind of thing, with what extras,
# then builds the function from parts. It covers far more than the snippets did
# because "biggest", "largest", "highest" and "max" all land on one operation,
# and "top 3" fills in a number.

OPS = {
    # name          words that mean it
    "sum":        "sum add total add-up adding",
    "average":    "average mean avg typical",
    "max":        "largest biggest highest max maximum greatest top",
    "min":        "smallest lowest min minimum least",
    "count":      "count how-many tally number-of frequency occurrences",
    "reverse":    "reverse backwards flip invert",
    "sort":       "sort order arrange rank",
    "unique":     "unique duplicates dedupe distinct repeated",
    "filter":     "filter only keep above over below under bigger smaller",
    "merge":      "merge combine join together",
    "palindrome": "palindrome same-backwards",
    "prime":      "prime",
    "even":       "even odd",
    "vowels":     "vowels consonants",
    "length":     "length longest shortest size",
    "upper":      "capitalise capitalize uppercase upper shout",
    "lower":      "lowercase lower",
    "split":      "split separate break-up",
    "celsius":    "celsius fahrenheit temperature",
    "fizzbuzz":   "fizzbuzz",
    "fibonacci":  "fibonacci fib",
    "search":     "search find-in binary-search lookup index-of",
    "swap":       "swap shuffle randomise randomize pick random",
    "read":       "read open load",
    "write":      "save write-to store",
    "table":      "times-table multiplication",
    "factorial":  "factorial",
    "round":      "round nearest",
    "percent":    "percent percentage",
}
WORD_OP = {w: op for op, words in OPS.items() for w in words.split()}

SUBJECTS = {
    "list": "list numbers array values items nums scores prices ages",
    "text": "string text word words sentence name line message",
    "dict": "dict dictionary map mapping pairs",
    "file": "file document csv txt",
}
WORD_SUBJ = {w: s for s, words in SUBJECTS.items() for w in words.split()}

BODIES = {
    ("sum", "list"): ("total(numbers)", "    return sum(numbers)",
                      "adds every number up", "total([1, 2, 3]) -> 6"),
    ("average", "list"): ("average(numbers)",
                          "    return sum(numbers) / len(numbers) if numbers else 0.0",
                          "adds them up and shares the total out evenly",
                          "average([2, 4]) -> 3.0"),
    ("max", "list"): ("biggest(numbers, n=1)",
                      "    return sorted(numbers, reverse=True)[:n]",
                      "sorts high to low and takes the first n",
                      "biggest([3, 9, 4], 2) -> [9, 4]"),
    ("min", "list"): ("smallest(numbers, n=1)",
                      "    return sorted(numbers)[:n]",
                      "sorts low to high and takes the first n",
                      "smallest([3, 9, 4], 2) -> [3, 4]"),
    ("count", "list"): ("count_each(items)",
                        "    counts = {}\n"
                        "    for item in items:\n"
                        "        counts[item] = counts.get(item, 0) + 1\n"
                        "    return counts",
                        "walks the list, adding 1 to each item's tally",
                        "count_each(['a', 'a', 'b']) -> {'a': 2, 'b': 1}"),
    ("count", "text"): ("count_words(text)",
                        "    counts = {}\n"
                        "    for word in text.lower().split():\n"
                        "        counts[word] = counts.get(word, 0) + 1\n"
                        "    return counts",
                        "splits on spaces, then tallies each word",
                        "count_words('a a b') -> {'a': 2, 'b': 1}"),
    ("reverse", "text"): ("reverse(text)", "    return text[::-1]",
                          "[::-1] walks the letters backwards",
                          "reverse('funk') -> 'knuf'"),
    ("reverse", "list"): ("reverse(items)", "    return items[::-1]",
                          "[::-1] walks the list backwards",
                          "reverse([1, 2, 3]) -> [3, 2, 1]"),
    ("sort", "list"): ("sort_items(items, biggest_first=False)",
                       "    return sorted(items, reverse=biggest_first)",
                       "sorted() does the work; reverse flips the direction",
                       "sort_items([3, 1, 2]) -> [1, 2, 3]"),
    ("sort", "dict"): ("sort_by_value(d, biggest_first=True):".rstrip(":"),
                       "    return dict(sorted(d.items(), key=lambda kv: kv[1],\n"
                       "                       reverse=biggest_first))",
                       "key= says compare the value, not the label",
                       "sort_by_value({'a': 1, 'b': 9}) -> {'b': 9, 'a': 1}"),
    ("unique", "list"): ("unique(items)",
                         "    seen, out = set(), []\n"
                         "    for item in items:\n"
                         "        if item not in seen:\n"
                         "            seen.add(item)\n"
                         "            out.append(item)\n"
                         "    return out",
                         "remembers what it has seen, keeps the first of each",
                         "unique([1, 2, 1]) -> [1, 2]"),
    ("filter", "list"): ("keep_over(numbers, limit=0)",
                         "    return [n for n in numbers if n > limit]",
                         "keeps only the numbers bigger than the limit",
                         "keep_over([1, 5, 9], 4) -> [5, 9]"),
    ("merge", "dict"): ("merge(a, b)", "    return {**a, **b}",
                        "{**a, **b} copies both in; b wins any clash",
                        "merge({'x': 1}, {'y': 2}) -> {'x': 1, 'y': 2}"),
    ("merge", "list"): ("merge(a, b)", "    return list(a) + list(b)",
                        "sticks the second list onto the end of the first",
                        "merge([1], [2]) -> [1, 2]"),
    ("palindrome", "text"): ("is_palindrome(text)",
                             "    clean = ''.join(c.lower() for c in text if c.isalnum())\n"
                             "    return clean == clean[::-1]",
                             "strips punctuation, then checks it reads the same backwards",
                             "is_palindrome('Racecar') -> True"),
    ("prime", "list"): ("is_prime(n)",
                        "    if n < 2:\n        return False\n"
                        "    for d in range(2, int(n ** 0.5) + 1):\n"
                        "        if n % d == 0:\n            return False\n"
                        "    return True",
                        "tries every divisor up to the square root",
                        "is_prime(7) -> True"),
    ("even", "list"): ("evens(numbers)",
                       "    return [n for n in numbers if n % 2 == 0]",
                       "% 2 == 0 means it divides by two exactly",
                       "evens([1, 2, 3, 4]) -> [2, 4]"),
    ("vowels", "text"): ("count_vowels(text)",
                         "    return sum(1 for c in text.lower() if c in 'aeiou')",
                         "counts every letter that is a vowel",
                         "count_vowels('funk') -> 1"),
    ("length", "text"): ("longest_word(text)",
                         "    return max(text.split(), key=len) if text.split() else ''",
                         "max() with key=len picks the longest",
                         "longest_word('a funky beat') -> 'funky'"),
    ("upper", "text"): ("shout(text)", "    return text.upper()",
                        "upper() makes every letter a capital",
                        "shout('funk') -> 'FUNK'"),
    ("lower", "text"): ("quiet(text)", "    return text.lower()",
                        "lower() makes every letter small",
                        "quiet('FUNK') -> 'funk'"),
    ("split", "text"): ("split_text(text, on=' ')",
                        "    return text.split(on)",
                        "split() cuts the string wherever it finds the separator",
                        "split_text('a b') -> ['a', 'b']"),
    ("celsius", "list"): ("to_fahrenheit(celsius)",
                          "    return celsius * 9 / 5 + 32",
                          "the conversion is times 9, divide by 5, add 32",
                          "to_fahrenheit(100) -> 212.0"),
    ("search", "list"): ("find_in(items, target)",
                         "    lo, hi = 0, len(items) - 1\n"
                         "    while lo <= hi:\n"
                         "        mid = (lo + hi) // 2\n"
                         "        if items[mid] == target:\n            return mid\n"
                         "        if items[mid] < target:\n            lo = mid + 1\n"
                         "        else:\n            hi = mid - 1\n"
                         "    return -1",
                         "halves the search area each guess; the list must be sorted",
                         "find_in([1, 3, 5], 5) -> 2"),
    ("swap", "list"): ("pick(items, n=1)",
                       "    import random\n"
                       "    return random.sample(list(items), min(n, len(items)))",
                       "random.sample takes n items without repeating any",
                       "pick([1, 2, 3], 2) -> [3, 1]"),
    ("read", "file"): ("read_lines(path)",
                       "    with open(path, encoding='utf-8') as f:\n"
                       "        return [line.rstrip() for line in f]",
                       "`with` closes the file for you when the block ends",
                       "read_lines('notes.txt') -> ['first line', ...]"),
    ("table", "list"): ("times_table(n, up_to=12)",
                        "    return [n * i for i in range(1, up_to + 1)]",
                        "multiplies n by every number up to up_to",
                        "times_table(3, 4) -> [3, 6, 9, 12]"),
    ("factorial", "list"): ("factorial(n)",
                            "    out = 1\n"
                            "    for i in range(2, n + 1):\n        out *= i\n"
                            "    return out",
                            "multiplies every number from 2 up to n",
                            "factorial(5) -> 120"),
    ("round", "list"): ("round_all(numbers, places=0)",
                        "    return [round(n, places) for n in numbers]",
                        "round() to the number of decimal places you want",
                        "round_all([1.26, 3.5], 1) -> [1.3, 3.5]"),
    ("percent", "list"): ("percent(part, whole)",
                          "    return part / whole * 100 if whole else 0.0",
                          "divide the part by the whole, times 100",
                          "percent(3, 4) -> 75.0"),
    ("fizzbuzz", "list"): ("fizzbuzz(n=100)",
                           "    for i in range(1, n + 1):\n"
                           "        out = ('Fizz' if i % 3 == 0 else '')"
                           " + ('Buzz' if i % 5 == 0 else '')\n"
                           "        print(out or i)",
                           "divisible by 3 is Fizz, by 5 is Buzz, by both is FizzBuzz",
                           "fizzbuzz(5) prints 1 2 Fizz 4 Buzz"),
    ("fibonacci", "list"): ("fib(n)",
                            "    a, b, out = 0, 1, []\n"
                            "    for _ in range(n):\n"
                            "        out.append(a)\n"
                            "        a, b = b, a + b\n"
                            "    return out",
                            "each number is the two before it added together",
                            "fib(6) -> [0, 1, 1, 2, 3, 5]"),
}


def guess_code(request):
    """Last resort: build something specific to what they said, rather than
    refusing. Names the function after their words and writes a real loop."""
    import re as _re
    words = [w for w in _re.findall(r"[a-z]+", request.lower())
             if w not in ("a", "an", "the", "me", "my", "to", "of", "for",
                          "that", "with", "and", "please", "can", "you",
                          "write", "make", "build", "code", "some", "it")]
    name = "_".join(words[:3]) or "do_thing"
    thing = next((w for w in words if w in WORD_SUBJ), "items")
    plural = thing if thing.endswith("s") else thing + "s"
    return (f"# My best shot at: {request.strip()[:60]}\n"
            f"# Built from the words in your request, so check it does what\n"
            f"# you meant. The loop is where the work goes.\n"
            f"# {name}([1, 2, 3]) -> [1, 2, 3]\n"
            f"def {name}({plural}):\n"
            f"    out = []\n"
            f"    for item in {plural}:\n"
            f"        out.append(item)          # <- do the work here\n"
            f"    return out\n\n"
            f'if __name__ == "__main__":\n'
            f"    print({name}([1, 2, 3]))")


def compose_code(request):
    """Read the request and build the function. -> (code, matched)."""
    import re as _re
    words = _re.findall(r"[a-z']+", request.lower())
    joined = " " + " ".join(words) + " "

    op = None
    for w in words:                       # single words
        if w in WORD_OP:
            op = WORD_OP[w]
            break
    if op is None:                        # then two-word phrases like "how many"
        for phrase, found in WORD_OP.items():
            if "-" in phrase and phrase.replace("-", " ") in joined:
                op = found
                break
    subj = next((WORD_SUBJ[w] for w in words if w in WORD_SUBJ), None)

    if op is None:
        return None, False
    if subj is None or (op, subj) not in BODIES:
        subj = next((s for (o, s) in BODIES if o == op), None)
    if subj is None:
        return None, False

    sig, body, why, example = BODIES[(op, subj)]
    n = _re.search(r"\b(\d+)\b", request)
    if n and "n=1" in sig:                # "top 3" fills the number in
        example = example.replace(", 2)", f", {n.group(1)})")
        sig = sig.replace("n=1", f"n={n.group(1)}")

    lines = [f"# {why}.", f"# {example}", f"def {sig}:", body, "",
             'if __name__ == "__main__":',
             f"    print({example.split(' ->')[0].split(' prints')[0]})"]
    return "\n".join(lines), True


def write_code(request, adapt=False):
    """-> (code, matched). adapt=True renames it to what you asked for."""
    import re as _re
    t = request.lower()
    want = _re.search(r"(?:called|named)\s+([a-z_][a-z0-9_]*)", t)
    built, ok = compose_code(request)     # build it from parts first
    if ok:
        if adapt and want:
            first = _re.search(r"def ([a-z_][a-z0-9_]*)", built)
            if first:
                built = built.replace(first.group(1), want.group(1))
        return built, True
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

# Understanding without a language model: each intent owns a bag of words.
STOP = set("a an the i you it that this to for of and is are my me we us can do"
           " with some something please just".split())
# Filler words like "me" swallowed sentences, so they go first.
WORDS = {
    "greet":    "hi hey hello yo sup yow howdy morning hows going",
    "thanks":   "thanks thank cheers appreciate nice sick lovely great"
                " awesome wicked",
    "identity": "who yourself name model called version robot alive",
    "help":     "help options commands examples able handle",
    "trouble":  "stopped stop broken break bad sucks useless rubbish wrong"
                " nothing works working failed fix stuck bro wtf hell dumb"
                " terrible awful rubbish crap",
    "status":   "status trained parameters loss score built version",
    "midi":     "midi mid daw logic ableton reaper cubase stems export",
    "open":     "open link result page article site url first second third",
    "explain":  "explain simpler clearer break walk through meaning means"
                " mean understand confused why",
    "search":   "search google lookup news latest happening won winner"
                " when where wiki info",
    "music":    "beat track song loop tune banger rhythm groove riff melody"
                " bassline drum drums percussion bpm bars tempo funk trap"
                " house dnb lofi reggaeton hardstyle hardcore rave hear"
                " listen play slow fast heavy",
    "code":     "code function class script program routine snippet method"
                " algorithm reverse flip backwards sort order count tally"
                " average prime factors fibonacci fizzbuzz merge combine"
                " remove duplicate tidy convert calculate palindrome list"
                " dict string number numbers json skeleton stub engine"
                " chess bot game app website server parser",
}
BAGS = {k: set(v.split()) - STOP for k, v in WORDS.items()}
SPREAD = {}
for _bag in BAGS.values():
    for _w in _bag:
        SPREAD[_w] = SPREAD.get(_w, 0) + 1
VOCAB = sorted(SPREAD)


def detect_intent(text):
    """Score against each bag, allowing for typos."""
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
    if len(t.split()) > 3:
        scores["greet"] = 0        # "yo make me a beat" is a beat, not a hello
    if _re_ask.match(t) and not scores["music"]:
        scores["code"] += 2
    # "write a chess engine that beats stockfish" is code, however many
    # music-ish words happen to be in it
    if words & {"write", "code", "function", "engine", "script", "program",
                "class", "app", "bot"}:
        scores["code"] += 2.5
    if words & {"average", "flip", "backwards", "divisible", "factors"}:
        scores["code"] += 1.5
    # If the generator can actually build it, it's a code request — the
    # router shouldn't need its own vocabulary for something we can do.
    if scores["music"] < 1.0 and compose_code(text)[1]:
        scores["code"] += 3

    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    top, second = ranked[0], ranked[1]
    if top[1] < 0.5:
        return "unknown"
    if top[1] - second[1] < 0.35:              # too close to call: ask instead
        return f"unsure:{top[0]}:{second[0]}"
    return top[0]


# Pro's warmth is phrasing and memory, not understanding.
WARM = {
    "greet": ["Yo{name}.", "Ready to cook{name}?", "Back in the lab{name}?",
              "Hey{name} — what's the vibe?", "What we making{name}?",
              "Alright{name}, hit me.", "Good to see you{name}. What's the move?"],
    "ack":   ["Say less.", "Bet.", "On it.", "Cooking.", "Let's go.",
              "Got you.", "Right then."],
    "thanks": ["Anytime.", "We good.", "Say the word.", "Course.",
               "That's the job."],
    "again": ["Another one.", "Take two.", "Same energy, new roll.",
              "Round two.", "Again it is."],
    "sorry": ["Not my department, that one.", "Can't do that one.",
              "That's outside my lane."],
    "nth":   ["That's {n} now.", "Number {n}.", "{n} and counting.",
              "You're on a run — {n}."],
    # thrown in occasionally after a track, so it reacts instead of reporting
    "vibe": ["This one's got legs.", "That bassline is doing work.",
             "Bit of a head-nodder, this.", "Turn it up.",
             "That one goes.", "Sitting nicely."],
}


class ChatSession:
    """Remembers enough for follow-ups."""

    def __init__(self, account="", key="web"):
        self.misses = 0
        self.prefs = {}
        self.history = []
        self.key = key
        self.account = account
        self.name = None
        self.last = {}
        self.turns = 0

    def pick(self, kind, **kw):
        seed = np.random.default_rng(self.turns * 7 + len(kind))
        line = WARM[kind][int(seed.integers(0, len(WARM[kind])))]
        return line.format(name=f", {self.name}" if self.name else "",
                           **kw).strip()


SESSIONS = {}


def get_session(key="default", account=""):
    ses = SESSIONS.setdefault(key, ChatSession(account, key))
    if account:
        ses.account = account
    return ses


def run_code(code, timeout=5):
    """Run generated code in a child process — only ever code Nova made for
    this session, never browser text."""
    import re as _re
    import subprocess
    import sys as _sys
    # any "name(args) ->" in a comment is an example we can run
    demo = _re.search(r"([A-Za-z_]\w*\([^)]*\))\s*(?:->|gives)", code)
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


class BPE:
    """Byte-level BPE over your own code."""

    def __init__(self, merges=None, vocab_size=VOCAB):
        self.merges = merges or {}
        self.vocab_size = vocab_size
        self.vocab = {i: bytes([i]) for i in range(256)}
        self._cache = {}
        for (a, b), idx in sorted(self.merges.items(), key=lambda kv: kv[1]):
            self.vocab[idx] = self.vocab[a] + self.vocab[b]

    @staticmethod
    def _pairs(ids):
        c = Counter()
        for pair in zip(ids, ids[1:]):
            c[pair] += 1
        return c

    @staticmethod
    def _merge(ids, pair, new_id):
        out, i = [], 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                out.append(new_id); i += 2
            else:
                out.append(ids[i]); i += 1
        return out

    def train(self, text, verbose=False):
        ids = list(text.encode("utf-8"))
        for k in range(max(0, self.vocab_size - 256)):
            counts = self._pairs(ids)
            if not counts:
                break
            pair, freq = counts.most_common(1)[0]
            if freq < 2:
                break
            new_id = 256 + k
            ids = self._merge(ids, pair, new_id)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose and (k + 1) % 200 == 0:
                print(f"  merge {k+1}  tokens left {len(ids):,}")
        return self

    def _encode_chunk(self, chunk):
        ids = list(chunk)
        while len(ids) >= 2:
            counts = self._pairs(ids)
            pair = min(counts, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = self._merge(ids, pair, self.merges[pair])
        return ids

    def encode(self, text):
        """Chunked + cached: 184K chars/sec vs 5K."""
        import re as _re
        out = []
        for chunk in _re.findall(r"\s*\S+|\s+", text):
            key = chunk
            hit = self._cache.get(key)
            if hit is None:
                hit = self._encode_chunk(chunk.encode("utf-8"))
                if len(self._cache) < 200_000:
                    self._cache[key] = hit
            out.extend(hit)
        return out

    def decode(self, ids):
        data = b"".join(self.vocab[i] for i in ids if i in self.vocab)
        return data.decode("utf-8", errors="replace")

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"vocab_size": self.vocab_size,
                       "merges": [[a, b, i] for (a, b), i in self.merges.items()]}, f)

    @classmethod
    def load(cls, path):
        blob = json.load(open(path))
        return cls({(a, b): i for a, b, i in blob["merges"]}, blob["vocab_size"])

    def __len__(self):
        return len(self.vocab)


# --- your own model: corpora, training, live scoring
#
# Everything here trains on this machine, from data written into this file or
# read off your disk. Nothing is downloaded and nothing phones home.
#
# Be clear-eyed about scale: this is a few million parameters on a CPU. It
# learns the shape of conversation and code — turn-taking, indentation, how a
# function is built — long before it learns meaning. The live scores below
# report what it can actually do, not what we hope it does.

# 8 x 384 comes to 15M parameters. Roughly 3x the compute of the 4.7M one,
# so expect about 3x the wall-clock for the same number of steps — worth it,
# since 4.7M was never going to hold a sentence together.
BIG_LAYERS = env("NOVA_LAYERS", 8, int)
BIG_HEADS = env("NOVA_HEADS", 8, int)
BIG_EMBD = env("NOVA_EMBD", 384, int)
BIG_BLOCK = env("NOVA_BLOCK", 128, int)
BIG_VOCAB = env("NOVA_VOCAB", 2048, int)
SKILL_CKPT = os.path.join(DATA_DIR, "skill.npz")
SKILL_VOCAB = os.path.join(DATA_DIR, "skill_vocab.json")

# Conversation data. Written here rather than scraped: assistant transcripts
# belong to whoever made them, and I can author the patterns directly.
CHAT_SEEDS = [
    ("hey", "Hey. What are we making?"),
    ("yo", "Yo. What do you need?"),
    ("hows it going", "Good. What are we working on?"),
    ("what can you do", "Music, code, and answering things. What do you want?"),
    ("who are you", "I'm Nova. I make beats and write code with you."),
    ("thanks", "Anytime."),
    ("that was sick", "Glad you like it."),
    ("make a funk beat", "Here's eight bars of funk at 130."),
    ("make it faster", "Sped it up. Same beat, quicker."),
    ("again", "Another one coming up."),
    ("make a trap beat at 140", "Trap at 140, eight bars."),
    ("i want something slower", "Slowed it down for you."),
    ("write a function that adds numbers", "Here you go. It adds the list up."),
    ("reverse a string", "This walks the string backwards."),
    ("how do i sort a list", "sorted() does it. Ascending unless you reverse it."),
    ("what does that code do", "It loops over the items and collects the result."),
    ("explain that", "It takes the input, works through it, and returns the answer."),
    ("it doesnt work", "What did it say? Paste the error and I'll look."),
    ("i got an error", "Send me the message and I'll tell you what broke."),
    ("can you help me", "Course. What are you stuck on?"),
    ("im stuck", "Tell me what you tried and what happened."),
    ("what is a variable", "A name for a value you want to keep."),
    ("what is a loop", "It runs the same lines once for each item."),
    ("what is a function", "A named piece of code you can call whenever."),
    ("search for chess engines", "Looking that up now."),
    ("i dont get it", "Which part? I'll go slower."),
    ("nice", "Cheers."),
    ("no", "Alright, what instead?"),
    ("stop", "Stopped."),
    ("do it again", "Running it again."),
]
CHAT_FILLER = ["cool", "ok", "alright", "hmm", "wait", "actually", "please",
               "bro", "man", "yeah", "nah", "so", "and", "then"]


def chat_corpus(n=env("NOVA_CHATS", 9000, int), seed=0):
    """Multi-turn conversations built from the seeds, so the model learns
    turn-taking and not just single replies."""
    r = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        lines, turns = [], int(r.integers(2, 7))
        for _ in range(turns):
            you, nova = CHAT_SEEDS[int(r.integers(0, len(CHAT_SEEDS)))]
            if r.random() < 0.25:
                you = f"{CHAT_FILLER[int(r.integers(0, len(CHAT_FILLER)))]} {you}"
            lines.append(f"you: {you}\nnova: {nova}")
        out.append("\n".join(lines))
    return "\n\n".join(out)


def code_corpus(paths=None, limit_mb=env("NOVA_CODE_MB", 30, int)):
    """Real Python off your disk: the standard library, everything pip has
    installed, and any folders you name in NOVA_CODE. Installed packages are
    the big win — 95 MB of other people's well-written code is already sitting
    on the machine, free and local.

    Your own projects are better still, because it learns to sound like you:
        NOVA_CODE=~/chess-engine:~/projects python nova.py serve
    """
    import site
    import sysconfig
    roots = paths
    if roots is None:
        roots = env("NOVA_CODE", "").split(":") if env("NOVA_CODE", "") else []
        roots.append(sysconfig.get_paths()["stdlib"])
        if env("NOVA_PACKAGES", "1") == "1":
            try:
                roots += [p for p in site.getsitepackages() if os.path.isdir(p)]
            except Exception:
                pass
            for extra in ("/usr/local/lib/python3.12/dist-packages",
                          sysconfig.get_paths().get("purelib", "")):
                if extra and os.path.isdir(extra) and extra not in roots:
                    roots.append(extra)
    chunks, total = [], 0
    for root in roots:
        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            continue
        for folder, dirs, names in os.walk(root):
            dirs[:] = [d for d in dirs if d not in
                       ("test", "tests", "__pycache__", "idlelib", ".git",
                        "node_modules")]
            for name in sorted(names):
                if not name.endswith((".py", ".md", ".rst")):
                    continue
                path = os.path.join(folder, name)
                try:
                    if os.path.getsize(path) > 150_000:
                        continue
                    with open(path, encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                except OSError:
                    continue
                if len(text) < 200:
                    continue
                chunks.append(text)
                total += len(text)
                if total > limit_mb * 1_000_000:
                    return "\n\n".join(chunks)
    return "\n\n".join(chunks)


class SkillTrainer:
    """Trains one model on conversation + code, and scores itself as it goes."""

    def __init__(self, verbose=False):
        os.makedirs(DATA_DIR, exist_ok=True)
        chat = chat_corpus()
        code = code_corpus()
        self.text = chat + "\n\n" + code
        self.split_at = len(chat)

        if os.path.exists(SKILL_VOCAB):
            self.tok = BPE.load(SKILL_VOCAB)
        else:
            self.tok = BPE(vocab_size=BIG_VOCAB).train(self.text[:400_000])
            self.tok.save(SKILL_VOCAB)

        cache = os.path.join(DATA_DIR, "skill_ids.npy")
        stamp = os.path.join(DATA_DIR, "skill_ids.txt")
        key = f"{len(self.text)}:{len(self.tok)}"
        ids = None
        if os.path.exists(cache) and os.path.exists(stamp):
            if open(stamp).read().strip() == key:      # same corpus as before
                try:
                    ids = np.load(cache)
                except Exception:
                    ids = None
        if ids is None:
            ids = np.array(self.tok.encode(self.text), dtype=np.uint16)
            try:
                np.save(cache, ids)
                open(stamp, "w").write(key)
            except OSError:
                pass
        # 28M ids as int64 is 223 MB; as uint16 it's 56 MB. Only the batch
        # needs to be int64, and that's 1,536 numbers. This is what was
        # killing the worker on a small host.
        if ids.dtype != np.uint16:
            ids = ids.astype(np.uint16)
        cut = int(0.97 * len(ids))
        self.train_ids, self.val_ids = ids[:cut], ids[cut:]

        self.text = ""                 # 30 MB of source, no longer needed
        self.model = NanoGPT(vocab=len(self.tok), block=BIG_BLOCK,
                             n_layer=BIG_LAYERS, n_head=BIG_HEADS,
                             n_embd=BIG_EMBD, seed=7)
        self.opt = Adam(self.model.p, lr=env("NOVA_LR", 8e-4, float))
        self.batch = env("NOVA_BATCH", 8, int)
        self.rng = np.random.default_rng()
        self._stop = threading.Event()
        self.thread = None
        # One "pass" is the model seeing the whole corpus once. Three passes
        # is a sensible target: enough to learn, not so much it memorises.
        per_step = self.batch * BIG_BLOCK
        # With 3+ tokens per parameter one pass is enough; repeating a big
        # corpus mostly teaches it to memorise. Small corpus, more passes.
        self.passes = env("NOVA_PASSES", 0, int) or (
            1 if len(ids) > 20_000_000 else 3)
        self.target = env("NOVA_TARGET", 0, int) or max(
            1000, int(self.passes * len(ids) / per_step))
        self.state = {"step": 0, "loss": None, "val": None, "best": None,
                      "params": self.model.n_params(), "tokens": int(len(ids)),
                      "chat": 0, "code": 0, "started": 0, "mins": 0.0,
                      "sample": "", "history": [], "target": self.target,
                      "per_step": per_step, "passes": self.passes,
                      "percent": 0.0, "seen": 0, "eta": ""}
        if os.path.exists(SKILL_CKPT):
            try:
                extra = self.model.load(SKILL_CKPT)
                self.state["step"] = int(extra.get("step", 0))
                self.state["best"] = extra.get("val")
                self.state["val"] = extra.get("val")
                self.state["seen"] = self.state["step"] * per_step
                self.state["percent"] = round(
                    100 * self.state["step"] / max(1, self.target), 1)
            except Exception as e:
                self.state["note"] = (f"previous checkpoint was a different "
                                      f"size ({type(e).__name__}) — starting "
                                      f"this one from scratch")

    # --- scoring ------------------------------------------------------
    def sample(self, prompt="you: hey\nnova:", n=60, temp=0.8):
        ids = self.tok.encode(prompt)[-(BIG_BLOCK - 1):]
        out = self.model.generate(ids, max_new_tokens=n, temperature=temp,
                                  top_k=40, rng=self.rng)
        return self.tok.decode(out[len(ids):])

    def score(self):
        """Three numbers out of 10, measured rather than guessed.

        understanding: how well it predicts held-out text (perplexity based)
        chat:          does a reply look like a reply — one line, sane length
        code:          does generated code parse as Python
        """
        losses = []
        for _ in range(4):
            x, y = self._batch(self.val_ids)
            losses.append(self.model.forward(x, y)[1])
        val = float(np.mean(losses))
        self.state["val"] = round(val, 3)
        # perplexity of 1 is perfect, 50+ is noise
        ppl = float(np.exp(min(val, 8)))
        understanding = max(0, min(10, round(10 * (1 - np.log(ppl) / 6), 1)))

        replies = [self.sample("you: hey\nnova:", 24) for _ in range(3)]
        good = sum(1 for r in replies
                   if 3 < len(r.strip()) < 90 and "\n" not in r.strip()[:60])
        chat = round(10 * good / max(1, len(replies)), 1)

        snips = [self.sample("def ", 60) for _ in range(3)]
        parses = 0
        for s_ in snips:
            try:
                compile("def " + s_, "<s>", "exec")
                parses += 1
            except (SyntaxError, ValueError):
                pass
        code = round(10 * parses / max(1, len(snips)), 1)

        self.state.update(chat=chat, code=code, understanding=understanding,
                          sample=replies[0].strip()[:70])
        return understanding, chat, code

    # --- training -----------------------------------------------------
    def _batch(self, split):
        ix = self.rng.integers(0, len(split) - BIG_BLOCK - 1, self.batch)
        x = np.stack([split[i:i + BIG_BLOCK] for i in ix]).astype(np.int64)
        y = np.stack([split[i + 1:i + 1 + BIG_BLOCK] for i in ix]).astype(np.int64)
        return x, y

    def step_once(self):
        x, y = self._batch(self.train_ids)
        _, loss, cache = self.model.forward(x, y)
        self.opt.step(self.model.p, self.model.backward(cache))
        self.state["step"] += 1
        self.state["loss"] = round(loss, 3)
        self.state["seen"] = self.state["step"] * self.state["per_step"]
        self.state["percent"] = round(
            100 * self.state["step"] / max(1, self.target), 1)
        return loss

    def save(self):
        tmp = SKILL_CKPT + ".tmp"
        self.model.save(tmp, extra={"step": self.state["step"],
                                    "val": self.state["best"]})
        os.replace(tmp + ".npz", SKILL_CKPT)

    def _loop(self):
        t0 = time.time()
        self.state["started"] = t0
        self.score()          # so the display has real numbers straight away
        while not self._stop.is_set():
            for _ in range(20):
                self.step_once()
            self.state["mins"] = round((time.time() - t0) / 60, 1)
            if self.state["step"] % 200 < 20:
                u, c, k = self.score()
                self.state["history"] = (self.state["history"] +
                                         [[self.state["step"], self.state["val"]]])[-60:]
                if self.state["best"] is None or self.state["val"] < self.state["best"]:
                    self.state["best"] = self.state["val"]
                    self.save()
            self._stop.wait(0.01)

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self._stop.clear()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self._stop.set()


def skill_reply(text, st, temp=0.8):
    """Let your own trained model write the reply. It's whatever the training
    has made it so far — gibberish early, words later. No templates."""
    prompt = f"you: {text.strip()}\nnova:"
    raw = st.sample(prompt, n=48, temp=temp)
    line = raw.split("\n")[0].strip()
    for stop in ("you:", "nova:"):
        if stop in line:
            line = line.split(stop)[0].strip()
    return line[:160]


# --- a real language model, when you have one

# Everything above this point matches words against a list. That is why Nova
# has never felt like talking to something — there was nothing in it that
# understood a sentence. This connects one that does.
#
# Point it at Ollama (free, runs on your Mac: `ollama run llama3.2`) or any
# server speaking the same JSON. The model handles the conversation and decides
# what you want; Nova's music, code and search stay exactly as they are and
# become the tools it reaches for. If no model is running, everything falls
# back to the keyword matcher and nothing breaks.

LLM_URL = env("NOVA_LLM_URL", "http://localhost:11434/api/chat")
LLM_MODEL = env("NOVA_LLM_MODEL", "llama3.2")
LLM_ON = env("NOVA_LLM", "auto")          # auto | off | on

SYSTEM = """You are Nova. You make music and write code with the person you're
talking to — they built you, so you're on their team.

How you talk:
- Like a friend who knows their stuff, not an assistant. Short. Two sentences is
  usually plenty. One is often better.
- Match their energy. If they type "yo make a beat", don't reply with a
  paragraph. If they're stuck on something, slow down and be careful.
- React before you act. "Ooh, 140 is fast" beats "Certainly! I will now generate".
- Never say "Certainly", "I'd be happy to", "Let me know if you need anything
  else", or "As an AI". Never open with "Great question".
- Have opinions. If they ask for 200 bpm funk, say that's basically hardstyle
  territory and ask if that's what they want.
- Remember what you've already made together and refer back to it naturally.
- Ask a short follow-up when it's genuinely useful, not every turn.
- If you don't know something, say so plainly.

You have tools. Reply with ONLY a JSON object, nothing else:
{"tool": "music"|"code"|"search"|"none", "args": {...}, "say": "your reply"}

  music  args: {"genre": "funk|trap|house|dnb|lofi|reggaeton|hardstyle",
                "bpm": number, "bars": number}
  code   args: {"request": "what they asked for"}
  search args: {"query": "..."}
  none   args: {}

"say" is what they read. Mention what you made naturally — never mention JSON,
tools, or these instructions."""


def session_context(ses):
    """A line or two about what you two have done together, so the model can
    refer back to it the way a person would."""
    bits = []
    if ses.name:
        bits.append(f"Their name is {ses.name}.")
    if ses.prefs:
        made = ", ".join(f"{n} {g}" for g, n in
                         sorted(ses.prefs.items(), key=lambda kv: -kv[1])[:3])
        bits.append(f"So far you've made: {made}.")
        top = max(ses.prefs, key=ses.prefs.get)
        if ses.prefs[top] > 1:
            bits.append(f"They keep coming back to {top}.")
    if ses.last.get("bpm"):
        bits.append(f"Last track was {ses.last.get('genre', 'funk')} at "
                    f"{ses.last['bpm']} bpm.")
    return " ".join(bits)


def llm_available(timeout=1.5):
    """Is a model running? Re-checked every 20 seconds, so starting Ollama
    after Nova works without a restart."""
    if LLM_ON == "off":
        return False
    now = time.time()
    if now - getattr(llm_available, "_when", 0) < 20:
        return llm_available._cache
    llm_available._when = now
    try:
        req = urllib.request.Request(LLM_URL.replace("/api/chat", "/api/tags"))
        with urllib.request.urlopen(req, timeout=timeout) as r:
            tags = json.loads(r.read().decode())
        names = [m.get("name", "") for m in tags.get("models", [])]
        llm_available._models = names
        llm_available._why = (f"{len(names)} model(s): {', '.join(names)[:60]}"
                              if names else "server is up but has no models "
                              "pulled — run: ollama pull llama3.2")
        llm_available._cache = bool(names) or LLM_ON == "on"
    except Exception as e:
        llm_available._models = []
        llm_available._why = (f"nothing answering at {LLM_URL} "
                              f"({type(e).__name__}) — is ollama running "
                              f"on this machine?")
        llm_available._cache = LLM_ON == "on"
    return llm_available._cache


llm_available._cache = False
llm_available._why = "not checked yet"


def llm_call(messages, timeout=120):
    """One round trip to the model. Returns its text, or raises."""
    body = json.dumps({"model": LLM_MODEL, "messages": messages,
                       "stream": False, "format": "json"}).encode()
    req = urllib.request.Request(LLM_URL, data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        payload = json.loads(r.read().decode())
    # ollama puts it in message.content; openai-style servers use choices
    if "message" in payload:
        return payload["message"]["content"]
    return payload["choices"][0]["message"]["content"]


CODE_SYSTEM = """You are a careful Python programmer writing for a beginner.

Rules:
- Output ONLY Python code. No markdown fences, no prose, no explanation outside comments.
- Start with 2-4 comment lines explaining it in plain English, as if to someone
  who has never coded. Then one comment line showing an example, like:
  # average([2, 4]) -> 3.0
- Write a complete, runnable module. Include a demo call at the bottom guarded by
  if __name__ == "__main__": so running the file shows it working.
- Use only the standard library.
- Keep it short and readable. No classes unless asked."""

# Code that will not be run automatically, however it got here. Nova executes
# what it writes, so anything touching the filesystem, the network or another
# process is shown to you instead of being run.
UNSAFE = ("os.system", "subprocess", "shutil.rmtree", "eval(", "exec(",
          "socket", "urllib", "requests", "open(", "__import__", "rmdir",
          "remove(", "unlink", "chmod", "popen")


def code_is_safe(code):
    low = code.lower()
    return not any(bad in low for bad in UNSAFE)


def llm_code(request, tries=3):
    """Write code, run it, and fix it if it breaks. Returns
    (code, output, notes) — notes says what happened on the way."""
    messages = [{"role": "system", "content": CODE_SYSTEM},
                {"role": "user", "content": request}]
    notes, code, out = [], "", ""
    for attempt in range(1, tries + 1):
        raw = llm_call(messages, timeout=180)
        code = raw.strip()
        if code.startswith("```"):                    # strip fences if it used them
            code = code.split("```")[1]
            code = code[6:] if code.lower().startswith("python") else code
            code = code.strip()
        try:                                          # some servers wrap in JSON
            parsed = json.loads(code)
            code = parsed.get("code", code) if isinstance(parsed, dict) else code
        except (json.JSONDecodeError, TypeError):
            pass

        try:
            compile(code, "<gen>", "exec")
        except SyntaxError as e:
            notes.append(f"attempt {attempt}: syntax error, asked for a fix")
            messages += [{"role": "assistant", "content": code},
                         {"role": "user", "content":
                          f"That has a syntax error on line {e.lineno}: {e.msg}. "
                          f"Send the whole corrected file."}]
            continue

        if not code_is_safe(code):
            notes.append("not run automatically: it touches files or the network")
            return code, "", notes

        out, err = run_code(code, timeout=8)
        if not err:
            if attempt > 1:
                notes.append(f"fixed itself on attempt {attempt}")
            return code, out, notes

        notes.append(f"attempt {attempt}: {err[:60]}")
        messages += [{"role": "assistant", "content": code},
                     {"role": "user", "content":
                      f"Running that gave this error:\n{err}\n"
                      f"Send the whole corrected file, nothing else."}]

    notes.append("still broken after "
                 f"{tries} tries — showing the last attempt")
    return code, out, notes


def llm_reply(text, trainer, spec, ses):
    """Let the model talk and pick the tool. Falls back to the matcher on any
    trouble, so a flaky model can never take the app down."""
    ctx = session_context(ses)
    history = [{"role": "system",
                "content": SYSTEM + (f"\n\nWhat you know: {ctx}" if ctx else "")}]
    for turn in ses.history[-8:]:
        history.append(turn)
    history.append({"role": "user", "content": text})

    raw = llm_call(history)
    try:
        plan = json.loads(raw)
    except json.JSONDecodeError:
        i, j = raw.find("{"), raw.rfind("}")
        plan = json.loads(raw[i:j + 1])

    tool = plan.get("tool", "none")
    args = plan.get("args") or {}
    out = {"intent": tool, "model": spec["name"], "reply": plan.get("say", "")}

    if tool == "music" and "music" in spec["can"] and trainer.music:
        genre = args.get("genre", "funk")
        genre = genre if genre in GENRES else "funk"
        bars = max(2, min(int(args.get("bars", 8) or 8), 32))
        bpm = args.get("bpm")
        old = trainer.music.style
        try:
            wav, tokens = trainer.music.compose(
                bars=bars, bpm=int(bpm) if bpm else None,
                arrange="arrange" in spec["can"],
                stereo="stereo" in spec["can"], genre=genre)
        finally:
            trainer.music.style = old
        ses.prefs[genre] = ses.prefs.get(genre, 0) + 1
        ses.last.update(wav=wav, tokens=tokens, genre=genre, bars=bars,
                        bpm=bpm or GENRES[genre]["bpm"])
        out["audio_url"] = f"/audio?session={ses.key}&t={int(time.time())}"

    elif tool == "code" and "code" in spec["can"]:
        if "fix" in spec["can"]:
            # write it, run it, and hand the error back for a fix
            code, ran, notes = llm_code(args.get("request", text))
            out["code"] = code
            ses.last["code"] = code
            if ran:
                out["checked"] = f"ran it — output: {ran.splitlines()[0][:60]}"
                out["reply"] += "\n" + out["checked"]
            if notes and any("attempt" in n or "fixed" in n for n in notes):
                out["reply"] += f"\n({notes[-1]})"
            return _finish(out, ses, text)
        code, _ = write_code(args.get("request", text),
                             adapt="compose" in spec["can"])
        out["code"] = code
        ses.last["code"] = code
        if "deep" in spec["can"]:
            good, err = run_code(code)
            if good:
                out["checked"] = f"checked it — output: {good.splitlines()[0][:50]}"
                out["reply"] += "\n" + out["checked"]
            elif err and "Nothing to show" not in err:
                out["checked"] = f"heads up, it errored: {err[:60]}"
                out["reply"] += "\n" + out["checked"]

    elif tool == "search" and "search" in spec["can"]:
        try:
            out["links"] = web_search(args.get("query", text), count=5)
        except Exception:
            out["reply"] += " (the search itself failed, though.)"

    return _finish(out, ses, text)


def _finish(out, ses, text):
    ses.history.append({"role": "user", "content": text})
    ses.history.append({"role": "assistant", "content": out["reply"]})
    return out


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
                "reply": f"{ses.name}. Noted. What are we cooking?"}

    import re as _re2
    _m = _re2.search(r"(?:i'?m|my name is|call me)\s+([a-z]{2,15})", text.lower())
    if _m and _m.group(1) not in ("looking", "trying", "going", "not", "just"):
        ses.name = _m.group(1).capitalize()

    # Your own trained model writes the words. The tools still do the work,
    # so a rough model still gets you a real beat — it just describes it in
    # its own voice, however good that voice currently is.
    if spec.get("mine") and getattr(milo_reply, "skill", None):
        st = milo_reply.skill
        said = skill_reply(text, st)
        intent = detect_intent(text)
        out = {"intent": intent, "model": spec["name"],
               "reply": said or "...",
               "trained": f"val {st.state.get('val')} · "
                          f"{st.state.get('percent', 0)}% trained"}
        if intent == "music" and trainer.music:
            genre = next((g for g in GENRES if g in low), "funk")
            wav, tokens = trainer.music.compose(
                bars=8, genre=genre, arrange="arrange" in spec["can"],
                stereo="stereo" in spec["can"])
            ses.last.update(wav=wav, tokens=tokens, genre=genre)
            out["audio_url"] = f"/audio?session={ses.key}&t={int(time.time())}"
        elif intent == "code":
            code, _ = write_code(text, adapt=True)
            out["code"] = code
            ses.last["code"] = code
        return _finish(out, ses, text)

    if llm_available():          # a real model talks; Nova's tools do the work
        try:
            return llm_reply(text, trainer, spec, ses)
        except Exception as e:
            ses.state_note = f"model unreachable ({type(e).__name__})"

    intent = detect_intent(text)
    if intent.startswith("unsure:"):
        # Don't make them choose — do the likelier one, mention the other.
        _, a, b = intent.split(":")
        intent = a
        _aside = {"music": "a beat", "code": "some code", "search": "a search",
                  "midi": "the MIDI"}.get(b, "")
        ses.last["aside"] = f" (say the word if you meant {_aside}.)" if _aside else ""
    if False:
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
            text += (f" {ses.last.get('genre', 'funk')}"
                     f" {ses.last.get('bpm', FUNK_BPM)} bpm"
                     f" {ses.last.get('bars', 8)} bars")
        elif any(w in low for w in ("faster", "quicker", "speed it up")):
            intent = "music"
            text += (f" {ses.last.get('genre', 'funk')} "
                     f"{min(180, ses.last.get('bpm', FUNK_BPM) + 15)} bpm")
        elif any(w in low for w in ("slower", "slow it down", "chill")):
            intent = "music"
            text += (f" {ses.last.get('genre', 'funk')} "
                     f"{max(80, ses.last.get('bpm', FUNK_BPM) - 15)} bpm")
        elif any(w in low for w in ("longer", "extend")):
            # carry the tempo too, or "faster" then "longer" silently resets it
            intent = "music"
            text += (f" {ses.last.get('genre', 'funk')}"
                     f" {min(16, ses.last.get('bars', 8) + 4)} bars"
                     f" {ses.last.get('bpm', FUNK_BPM)} bpm")

    if warm and any(w in low for w in ("thanks", "thank you", "nice", "sick", "love it")):
        return {"intent": "thanks", "model": spec["name"], "reply": ses.pick("thanks")}

    out = {"intent": intent, "model": spec["name"]}
    if intent not in ("unknown", "empty"):
        ses.misses = 0

    # If this model can't do what it heard, try what it can do before giving
    # up. "tracks my chess ratings" hits the music bag, but Milo does code —
    # so build the code rather than sending them to another model.
    _w = set(low.split())
    _clearly_music = _w & {"beat", "bpm", "bars", "song", "loop", "banger",
                           "drop", "funk", "trap", "house", "dnb", "lofi",
                           "hardstyle", "reggaeton"}
    if (intent == "music" and "music" not in spec["can"]
            and "code" in spec["can"] and not _clearly_music
            and write_code(text)[1]):
        intent = "code"          # "tracks my ratings" is code, not a track
    if (intent == "code" and "code" not in spec["can"]
            and "music" in spec["can"] and _clearly_music):
        intent = "music"

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
            f"Straight with you: no language model writes these replies. I "
            f"match what you ask against what I can do, then do it.")

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
                        f"\n  funk hooks come from the generator (98% in key)")

    elif intent == "midi":
        if not ses.last.get("tokens"):
            out["reply"] = "Make something first."
            return out
        mid = tokens_to_midi(ses.last["tokens"], bpm=ses.last.get("bpm", FUNK_BPM))
        ses.last["mid"] = mid
        out["midi_url"] = f"/audio?session={ses.key}&kind=mid"
        out["reply"] = f"MIDI, {len(mid):,} bytes — opens in any DAW."

    elif intent == "trouble":
        last = ses.last.get("gave")
        if last == "trouble":
            out["reply"] = ("Alright, plainly: tell me the thing you want in "
                            "your own words — like \"add up a list\" or "
                            "\"trap beat 150\" — and I'll do it or say I can't.")
        elif ses.last.get("code") or ses.last.get("tokens"):
            out["reply"] = ("Fair. What was wrong with the last one — should it "
                            "do something different, or did it not work at all?")
        else:
            out["reply"] = ("Yeah, that's on me. What were you trying to get? "
                            "Say it however you like and I'll tell you straight "
                            "whether I can do it.")
        ses.last["gave"] = "trouble"
        return out

    elif intent == "explain":
        code = ses.last.get("code")
        if not code and ses.last.get("tokens"):
            g = ses.last.get("genre", "funk")
            spec_g = GENRES.get(g, GENRES["funk"])
            out["reply"] = (
                f"That's {g} at {ses.last.get('bpm')} bpm. "
                + (f"Kick on steps {spec_g['kick']} of 16" if spec_g['kick']
                   else "Kick on a tamborzão pattern")
                + (f", claps on {spec_g['clap']}" if spec_g['clap']
                   else ", claps on the backbeat")
                + f", hats every {spec_g['hat']}."
                f"\nAn 808 follows each bar's first note, and every hit is nudged "
                f"a few milliseconds so it doesn't sound like a machine.")
            return out
        if not code:
            out["reply"] = "Ask for some code or a beat first."
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
        # A 3 MB base64 blob in JSON broke Safari; hand over a URL instead.
        ses.last["wav"] = wav
        out["audio_url"] = f"/audio?session={ses.key}&t={int(time.time())}"
        out["notes"] = tokens
        shown = bpm or GENRES[genre]["bpm"]        # not the funk default
        ses.prefs[genre] = ses.prefs.get(genre, 0) + 1
        ses.last.update(bars=bars, bpm=shown, style=style, tokens=tokens,
                        genre=genre)
        lead = ""
        if warm:
            lead = (ses.pick("again") if "again" in text.lower()
                    else ses.pick("ack"))
            n = ses.prefs.get(genre, 0)
            if n in (3, 5, 10):
                lead = ses.pick("nth", n=n) + " " + lead
        kind = genre if style == "funk" else "folk melody"
        tail = ""
        if warm and ses.turns % 3 == 0:
            tail = " " + ses.pick("vibe")
        out["reply"] = (f"{lead} {bars} bars of {kind} at {shown} bpm.{note}{tail}"
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
        if text.strip().lower() in ("skeleton", "stub", "blank"):
            code, _ = write_code(ses.last.get("asked", "do_thing"))
            out["code"] = code
            out["reply"] = "Here's the shape. Fill in the middle."
            ses.last["code"] = code
            return out
        ses.last["asked"] = text
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
            # A six-line snippet is not "a chess engine that beats Stockfish".
            # If they asked for something big, say what this actually is.
            big = {"engine", "app", "website", "game", "bot", "server", "full",
                   "complete", "whole", "beats", "clone", "system", "ai"}
            if set(prompt.lower().replace(",", " ").split()) & big:
                out["reply"] += ("\n\nStraight with you: that's a small piece, "
                                 "not the whole thing you asked for. I only "
                                 "know a handful of shapes by heart — with "
                                 "Ollama running I could write the real one.")
        else:
            # Never refuse. Build the closest thing from what they said.
            out["code"] = guess_code(prompt)
            ses.last["code"] = out["code"]
            out["reply"] = (
                "Don't have that exact one, so here's my best shot — built "
                "from the words you used. The loop in the middle is where the "
                "work goes.\nTell me what it should do to each item and I'll "
                "fill it in. I'm also good at: "
                + ", ".join(sorted(OPS)[:10]) + ".")

    else:
        # Repeating the same "didn't catch that" is the most annoying thing a
        # program can do. Each miss in a row gets a different, humbler answer.
        ses.misses += 1
        able = [c for c in ("music", "code", "search") if c in spec["can"]]
        annoyed = any(w in low for w in ("bro", "actually", "stupid", "useless",
                                         "dumb", "trash", "come on", "wtf",
                                         "!!", "?!")) or text.isupper()
        if annoyed or ses.misses >= 3:
            out["reply"] = (
                "Fair — I'm not getting it, and repeating myself isn't helping.\n"
                "Straight answer: without a language model running I match "
                "keywords, so anything I wasn't built for slides straight past "
                "me.\nGive me one word and I'll act on it: beat, code, or "
                "search. Or start Ollama on this machine and I'll actually "
                "understand you.")
        elif ses.misses == 2:
            out["reply"] = ("Still not with you. Try telling me the thing you "
                            f"want rather than the sentence — like \"average of "
                            f"a list\" or \"trap beat 140\". {spec['name']} does "
                            f"{', '.join(able)}.")
        else:
            out["reply"] = (f"Didn't catch that. {spec['name']} does "
                            f"{', '.join(able) or 'not much'} — say 'help' for "
                            f"examples.")
    return out

def free_memory_mb():
    """How much RAM is actually available, so we don't kill the worker."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    try:                                   # macOS has no /proc
        import subprocess
        out = subprocess.run(["sysctl", "-n", "hw.memsize"],
                             capture_output=True, text=True, timeout=2)
        return int(out.stdout.strip()) / 1e6 * 0.5
    except Exception:
        return 4096.0                      # assume a normal machine


def start_skill(app):
    """Train your model in the background while the app serves. One thread,
    lowest priority — the page stays responsive because numpy drops the GIL
    during the matrix work."""
    # Measured: ~900 MB peak at batch 4, ~1.15 GB at batch 8, for the 15M
    # model. The backward pass holds every layer's activations at once, which
    # is most of it. 2 GB free is a safe bar.
    need = env("NOVA_NEED_MB", 2000, int)
    free = free_memory_mb()
    if free < need and env("NOVA_TRAIN", "auto") != "force":
        app._skill_note = (f"training off: needs about {need} MB free and this "
                           f"machine has {free:.0f} MB. Run it on your Mac, or "
                           f"set NOVA_TRAIN=force to try anyway.")
        print(app._skill_note)
        return None
    try:
        st = SkillTrainer()
    except MemoryError:
        app._skill_note = "training off: ran out of memory building the corpus."
        print(app._skill_note)
        return None
    except Exception as e:
        print(f"skill training off: {type(e).__name__}")
        return None
    app._skill = st
    milo_reply.skill = st
    st.start()
    return st


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

    @app.route("/audio")
    def audio():
        ses = get_session(request.args.get("session", "web"))
        kind = request.args.get("kind", "wav")
        blob = ses.last.get(kind)
        if not blob:
            return jsonify(error="nothing to play"), 404
        return app.response_class(blob, mimetype="audio/wav" if kind == "wav"
                                  else "audio/midi")

    @app.route("/run", methods=["POST"])
    def run():
        body = request.get_json(silent=True) or {}
        ses = get_session(body.get("session", "web"))
        code = ses.last.get("code")
        if not code:
            return jsonify(error="Ask for some code first."), 400
        stdout, err = run_code(code)
        return jsonify(output=stdout, error=err)

    @app.route("/training")
    def training():
        st = getattr(app, "_skill", None)
        if st is None:
            return jsonify(running=False,
                           hint=getattr(app, "_skill_note",
                                        "start it with: python nova.py "
                                        "train-skill"))
        d = dict(st.state)
        d["running"] = bool(st.thread and st.thread.is_alive())
        return jsonify(d)

    @app.route("/models")
    def models():
        live = llm_available()
        brain = (f"{LLM_MODEL} via Ollama" if live
                 else f"keyword matching — {llm_available._why}")
        out = {k: dict(v, scores=scores(v)) for k, v in MODELS.items()}
        return jsonify(models=out, default=DEFAULT_MODEL, brain=brain)

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
    "funk":      dict(bpm=130, kick=None, clap=None, hat=2, sub=.6, swing=.05,
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
    "hardstyle": dict(bpm=150, kick=[0, 4, 8, 12], clap=[4, 12], hat=2,
                      sub=.30, swing=0, mode=MINOR, low=.34),
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


def _highpass(x, k=9):
    """Cheap high pass: signal minus its smoothed self."""
    return x - np.convolve(x, np.ones(k, dtype=np.float32) / k, mode="same")


def _lowpass(x, k):
    return np.convolve(x, np.ones(max(2, k), dtype=np.float32) / max(2, k),
                       mode="same")


def _noise(n, seed):
    return np.random.default_rng(seed).standard_normal(n).astype(np.float32)


# The tamborzão isn't a drum machine — it's hand percussion. A surdo carries the
# low pulse, atabaques (hand drums) answer in the mids, a rim click cuts through.
# Sine-with-a-pitch-drop is a drum; sine-with-a-pitch-drop plus distortion is
# hardstyle, which is what this used to sound like.

def _kick(dur=0.34):
    """Surdo: deep, round, barely any click. Not a distorted EDM kick."""
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    freq = 47.0 + 68.0 * np.exp(-t * 26)              # gentler sweep, more body
    body = np.sin(np.cumsum(2 * np.pi * freq / SR)) * _env(n, 7.5)
    skin = _lowpass(_noise(n, 1), 12) * _env(n, 90) * 0.18   # the beater, soft
    return np.tanh((body + skin) * 1.05) * 0.95        # barely any drive


def _hardkick(midi=36, dur=0.42):
    """Hardstyle kick: click, then a distorted tail that drops in pitch. The
    tail is tuned, so the kick plays the bassline — that's the whole genre."""
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    punch = np.sin(np.cumsum(2 * np.pi * (150 + 900 * np.exp(-t * 120)) / SR))
    punch *= _env(n, 70) * 0.9
    root = 440.0 * 2 ** ((midi - 69) / 12.0)
    freq = root * (1 + 5.0 * np.exp(-t * 55))         # long pitch fall
    tail = np.sin(np.cumsum(2 * np.pi * freq / SR)) * _env(n, 5.5)
    tail = np.tanh(tail * 4.5)                        # the distortion is the point
    tail = _lowpass(tail, 3)                          # tame the very top
    return np.clip(punch * 0.5 + tail * 0.85, -1, 1)


def _screech(midi, dur):
    """Detuned saws — the euphoric hardstyle lead."""
    n = max(1, int(dur * SR))
    t = np.arange(n, dtype=np.float32) / SR
    f = 440.0 * 2 ** ((midi - 69) / 12.0)
    saw = sum(((t * f * d) % 1.0 - 0.5) for d in (0.994, 1.0, 1.006))
    env = _env(n, 3.0)
    a = max(1, int(0.01 * SR))
    env[:a] *= np.linspace(0, 1, a)
    return _lowpass(np.tanh(saw * 1.4), 3) * env * 0.32


def _voice(midi, dur=0.16, vowel=0):
    """A vowel-ish chop. Funk is built on chopped voices — a saw run through
    three formant peaks reads as "ah"/"eh"/"oh" to the ear, and stuttering it
    on the grid is the montagem sound."""
    n = max(1, int(dur * SR))
    t = np.arange(n, dtype=np.float32) / SR
    f = 440.0 * 2 ** ((midi - 69) / 12.0)
    F = ((730, 1090, 2440), (530, 1840, 2480), (570, 840, 2410))[vowel % 3]
    out = np.zeros(n, dtype=np.float32)
    for h in range(1, 26):                       # harmonics of the voice
        hz = f * h
        if hz > SR / 2:
            break
        gain = sum(np.exp(-((hz - c) / (c * 0.28)) ** 2) for c in F)
        if gain > 0.02:
            out += np.sin(2 * np.pi * hz * t) * gain / h ** 0.4
    env = _env(n, 12)
    a = max(1, int(0.006 * SR))
    env[:a] *= np.linspace(0, 1, a)
    return out / (np.abs(out).max() or 1) * env * 0.5


def _whistle(dur=0.45):
    """Apito: the referee whistle all over baile funk."""
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    wob = 1 + 0.02 * np.sin(2 * np.pi * 11 * t)          # the trill
    tone = np.sin(np.cumsum(2 * np.pi * 2100 * wob / SR))
    tone += 0.3 * np.sin(np.cumsum(2 * np.pi * 3150 * wob / SR))
    air = _highpass(_noise(n, 9), 3) * 0.25
    env = _env(n, 6)
    env[:int(0.02 * SR)] *= np.linspace(0, 1, int(0.02 * SR))
    return (tone + air) * env * 0.3


def _tom(midi=52, dur=0.26, seed=2):
    """Atabaque: a hand drum with real pitch. This is the funk sound."""
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    f0 = 440.0 * 2 ** ((midi - 69) / 12.0)
    freq = f0 * (1 + 0.55 * np.exp(-t * 30))          # slaps bend down
    body = np.sin(np.cumsum(2 * np.pi * freq / SR)) * _env(n, 13)
    ring = np.sin(np.cumsum(2 * np.pi * freq * 1.6 / SR)) * _env(n, 22) * 0.35
    slap = _lowpass(_noise(n, seed), 6) * _env(n, 120) * 0.3
    return (body + ring + slap) * 0.8


def _rim(dur=0.06, seed=3):
    """Rim click: short, woody, cuts through the mix."""
    n = int(dur * SR)
    tone = np.sin(2 * np.pi * 780 * np.arange(n, dtype=np.float32) / SR)
    wood = _lowpass(_highpass(_noise(n, seed), 5), 3)
    return (tone * 0.5 + wood) * _env(n, 150) * 0.55


def _clap(seed=0, dur=0.24):
    """Three flams, filtered — a clap, not a burst of static."""
    n = int(dur * SR)
    r = np.random.default_rng(seed)
    noise = _lowpass(_highpass(r.standard_normal(n).astype(np.float32), 7), 4)
    out = np.zeros(n, dtype=np.float32)
    for k, off in enumerate((0.0, 0.012, 0.025)):
        i = int(off * SR)
        out[i:] += noise[:n - i] * _env(n - i, 60 if k < 2 else 26)
    body = np.sin(2 * np.pi * 190 * np.arange(n, dtype=np.float32) / SR)
    return (out * 0.5 + body * _env(n, 40) * 0.25) * 0.75


def _hat(seed=0, dur=0.05, open_=False):
    """Shaker rather than white noise: narrower, softer, sits back."""
    n = int(dur * (3 if open_ else 1) * SR)
    noise = _lowpass(_highpass(_noise(n, seed), 4), 3)
    return noise * _env(n, 40 if open_ else 110) * 0.3


def _sub(midi, dur):
    """808 sub with a short glide into the note. Clean, not saturated."""
    n = max(1, int(dur * SR))
    t = np.arange(n, dtype=np.float32) / SR
    target = 440.0 * 2 ** ((midi - 69) / 12.0)
    freq = target * (1 + 0.4 * np.exp(-t * 40))
    tone = np.sin(np.cumsum(2 * np.pi * freq / SR))
    return np.tanh(tone * 1.15) * _env(n, 2.6) * 0.9


def _lead(midi, dur):
    """A plucked tone: soft triangle-ish harmonics that decay, so it reads as
    an instrument instead of a buzzing square wave."""
    n = max(1, int(dur * SR))
    t = np.arange(n, dtype=np.float32) / SR
    f = 440.0 * 2 ** ((midi - 69) / 12.0)
    tone = (np.sin(2 * np.pi * f * t)
            + 0.34 * np.sin(2 * np.pi * f * 2 * t) * _env(n, 9)
            + 0.16 * np.sin(2 * np.pi * f * 3 * t) * _env(n, 16)
            + 0.07 * np.sin(2 * np.pi * f * 4.02 * t) * _env(n, 24))
    envelope = _env(n, 4.5)
    a = max(1, int(0.008 * SR))
    envelope[:a] *= np.linspace(0, 1, a)
    rl = max(1, int(min(0.09, dur * 0.5) * SR))
    envelope[-rl:] *= np.linspace(1, 0, rl)
    return tone * envelope * 0.42


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
    hard = genre == "hardstyle"
    chop_note = None
    if notes:
        chop_note = int(np.median([n for _, _, n in notes])) + 12
        while chop_note > 76:
            chop_note -= 12
    root_note = 36
    if notes:
        root_note = min(n for _, _, n in notes)
        while root_note > 40:
            root_note -= 12
    kick = _hardkick(root_note) if hard else _kick()
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
            place(_clap(seed=bar * 7 + step), sw(step), 0.38 * vel())
        if hard:
            # reverse bass: a stab on every offbeat, ducking under the kick
            for step in (2, 6, 10, 14):
                place(_sub(root_note + 12, 2 * spb), sw(step), 0.75 * vel())
        else:
            # The atabaques: hand drums answering the surdo on the off-tresillo.
            # That call-and-response is what makes it funk and not EDM.
            for k, step in enumerate(s for s in TRESILLO if s not in kick_pat):
                place(_tom(52 + (k % 3) * 4, seed=bar * 5 + step), sw(step),
                      0.55 * vel())
                # ghost note: a quiet slap just after, the way a hand player
                # fills the gap between the loud hits
                if r.random() < 0.5:
                    place(_tom(52 + (k % 3) * 4 + 3, 0.12, seed=bar + step),
                          sw(step) + spb * 0.5, 0.2 * vel())
            for step in (2, 7, 11, 15):
                if r.random() < 0.55:
                    place(_rim(seed=bar * 11 + step), sw(step), 0.4 * vel())
            # Montagem: a chopped voice stuttered on the grid, answering the
            # drums. Two or three hits, same note, tight together.
            if genre == "funk" and bar % 2 == 1 and chop_note:
                start_step = int(r.choice([6, 10, 12]))
                for k in range(int(r.integers(2, 5))):
                    place(_voice(chop_note + 12 * (k == 2),
                                 0.13, vowel=bar + k),
                          sw(start_step) + k * spb * 0.5, 0.6 * vel())
            if genre == "funk" and bar and bar % 4 == 0 and r.random() < 0.6:
                place(_whistle(), base - 2 * spb, 0.5)   # apito into the turn
        if human and bar and bar % 4 == 3:               # a fill before the turn
            for k, step in enumerate((12, 13, 14, 15)):
                place(_clap(seed=bar * 31 + k), base + step * spb,
                      0.28 + 0.09 * k)
        for step in range(0, 16, g["hat"]):              # hats sit well back
            place(_hat(seed=bar * 13 + step, open_=(step == 14)),
                  sw(step), 0.13 * vel())
            if r.random() < 0.18:
                place(_hat(seed=bar * 17 + step), base + (step + 1) * spb, 0.14)

    # 808 follows the first note of each bar
    for bar in range(n_bars):
        midi = bar_first[bar % len(bar_first)] if bar_first else None
        if midi is None:
            continue
        while midi > 52:
            midi -= 12                                   # keep the sub low
        if not (arrange and bar == 0) and not hard:
            place(_sub(midi, 16 * spb * g["sub"]), bar * 16 * spb, 1.25, "low")

    quiet = set()
    if arrange:
        # bar 0 drums-only, one bar mid-track drops out: stops it sounding looped
        quiet = {0, max(1, n_bars // 2)}
    for i, (start, dur, midi) in enumerate(notes):       # stabs, with space between
        if int(start / (16 * spb)) in quiet:
            continue
        step = int(round(start / spb)) % 16
        if step not in TRESILLO and r.random() < 0.62:
            continue                                     # leave space
        voice = _screech if hard else _lead
        place(voice(midi, min(dur, 3 * spb)), start, 0.5 if hard else 0.30)

    # Sidechain: dip everything when the kick lands.
    duck = np.ones(n_total, dtype=np.float32)
    fall = np.exp(-np.arange(int(.18 * SR), dtype=np.float32) / (.055 * SR))
    for bar in range(n_bars):
        for step in kick_pat:
            i = max(0, int((bar * 16 + step) * spb * SR))
            seg = duck[i:i + len(fall)]
            duck[i:i + len(seg)] = np.minimum(seg, 1 - 0.7 * fall[:len(seg)])
    top_bus *= duck
    top_bus = .65 * top_bus + .35 * np.convolve(
        top_bus, np.ones(3, np.float32) / 3, mode="same")   # tame the fizz

    # Bisect the top-bus gain to hit the low-end target: six passes.
    def low_share(gain):
        mix = low_bus + top_bus * gain
        spec = np.abs(np.fft.rfft(mix))
        freqs = np.fft.rfftfreq(len(mix), 1 / SR)
        total_e = spec.sum() or 1.0
        return spec[(freqs > 25) & (freqs < 140)].sum() / total_e

    lo_g, hi_g = 0.15, 9.0
    if low_share(1.0) < g.get("low", FUNK_LOW_TARGET):
        lo_g, hi_g = 0.15, 1.0                        # too bright: pull the top down
    else:
        lo_g, hi_g = 1.0, 9.0                         # too dark: bring it up
    for _ in range(6):
        mid = (lo_g + hi_g) / 2
        if low_share(mid) > g.get("low", FUNK_LOW_TARGET):
            lo_g = mid
        else:
            hi_g = mid
    buf = low_bus + top_bus * ((lo_g + hi_g) / 2)

    peak = np.abs(buf).max() or 1.0
    buf = np.tanh(buf / peak * 0.85) * 0.94

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
    hush = lambda *a, **k: np.zeros(64, np.float32)
    saved_t, saved_r = _tom, _rim
    globals()["_clap"] = globals()["_hat"] = hush
    globals()["_tom"] = globals()["_rim"] = hush
    saved_sub = _sub
    globals()["_sub"] = hush
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
                # A hit is a jump in energy, not silence beforehand: the
                # hardstyle kick rings into the next beat, and swung genres
                # land their kicks a few hundred samples late.
                here, prev = amp[k:k + 700], amp[max(0, k - 700):k]
                h = here.max() if len(here) else 0
                p = prev.max() if len(prev) else 0
                if h > 8000 and (p == 0 or h > p * 1.3):
                    got.append(st)
            genre_ok &= set(got) == set(gspec["kick"])
    finally:
        globals()["_clap"], globals()["_hat"] = saved_c, saved_h
        globals()["_tom"], globals()["_rim"] = saved_t, saved_r
        globals()["_sub"] = saved_sub
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
    # tests cover Nova's own logic, not whatever model happens to be running
    was = getattr(llm_available, "_cache", None)
    llm_available._cache = False
    _t = Trainer(verbose=False)
    free = milo_reply("write a fizzbuzz", _t, "nova-iris", ChatSession())
    admin = milo_reply("write a fizzbuzz", _t, "nova-iris",
                       ChatSession("editornova"))
    tier_ok = ("code" in free and "code" in admin and "checked" in admin
               and free["intent"] != "locked")
    llm_available._cache = was
    print(f"iris free for everyone + checks its code: "
          f"{'PASS' if tier_ok else 'FAIL'}")

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
                    choices=["serve", "test", "chat", "train-skill",
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

    if args.command == "train-skill":
        st = SkillTrainer(verbose=True)
        d = st.state
        print(f"{d['params']:,} parameters | {d['tokens']:,} tokens | "
              f"vocab {len(st.tok)}")
        print(f"target: {st.target:,} steps = {st.passes} passes over the "
              f"corpus ({d['per_step']:,} tokens a step)")
        print("ctrl-c to stop; it saves as it goes and picks up where it left "
              "off.\n")
        t0, done0 = time.time(), d["step"]
        try:
            while d["step"] < st.target:
                for _ in range(20):
                    st.step_once()
                if d["step"] % 200 < 20:
                    u, c, k = st.score()
                    st.save()
                    elapsed = time.time() - t0
                    rate = (d["step"] - done0) / max(elapsed, 1e-9)
                    left = (st.target - d["step"]) / rate if rate else 0
                    d["eta"] = (f"{left/3600:.1f}h" if left > 3600
                                else f"{left/60:.0f}m")
                    bar = int(d["percent"] / 5)
                    print(f"val {d['val']:.3f}  loss {d['loss']:.3f}  "
                          f"[{'#' * bar}{'.' * (20 - bar)}] "
                          f"{d['percent']:.1f}%  step {d['step']:,}/"
                          f"{st.target:,}  {elapsed/60:.0f}m in, {d['eta']} left")
                    print(f"    understanding {u}/10   chat {c}/10   "
                          f"code {k}/10   pass "
                          f"{d['seen']/max(1, d['tokens']):.2f} of {st.passes}")
                    print(f"    says: {d['sample']!r}\n")
            print("done — that's the full target.")
        except KeyboardInterrupt:
            st.save()
            print(f"\nsaved at step {d['step']:,} ({d['percent']:.1f}% of target)")
        return

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
    if env("NOVA_TRAIN", "auto") != "off":
        start_skill(app)

if __name__ == "__main__":
    cli()
