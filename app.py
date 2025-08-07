import io
import sys
from typing import List, Tuple

import streamlit as st

# Optional deps: tiktoken (OpenAI), transformers+tokenizers (HF)
try:
    import tiktoken
except Exception:
    tiktoken = None

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None


# -------- Utilities --------
def tokenize_with_tiktoken(text: str, enc_name: str) -> Tuple[List[int], List[str]]:
    enc = tiktoken.get_encoding(enc_name)
    ids = enc.encode(text)
    # Decode each token individually to show boundaries
    tokens = [enc.decode([tid]) for tid in ids]
    return ids, tokens

@st.cache_resource(show_spinner=False)
def get_hf_tokenizer(model_name: str):
    if AutoTokenizer is None:
        raise RuntimeError("transformers is not installed")
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)

def tokenize_with_hf(text: str, model_name: str) -> Tuple[List[int], List[str]]:
    tok = get_hf_tokenizer(model_name)
    out = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = out["input_ids"]
    offsets = out["offset_mapping"]
    # Extract string spans by offsets
    tokens = [text[o[0]:o[1]] for o in offsets]
    return ids, tokens

def token_bytes(s: str) -> str:
    # Display underlying UTF-8 bytes as hex (e.g., e2 82 ac)
    return " ".join([f"{b:02x}" for b in s.encode("utf-8", errors="replace")])

def pretty_boundary(tokens: List[str]) -> str:
    # Show token boundaries as |token|
    if not tokens:
        return ""
    return "│" + "│".join(t.replace("\n", "\\n") for t in tokens) + "│"


# -------- UI --------
st.set_page_config(page_title="LLM Tokenization Explorer", layout="wide")
st.title("🔤 LLM Tokenization Explorer")

st.caption(
    "Explore how different tokenizers split text. View token IDs, per-token bytes, and compare two tokenizers side-by-side."
)

with st.sidebar:
    st.header("⚙️ Settings")

    demo_texts = {
        "English sentence": "The quick brown fox jumps over 13 lazy dogs.",
        "German (Umlauts)": "Falsches Üben von Xylophonmusik quält jeden größeren Zwerg.",
        "Emojis": "I love 🍕 and 🐍! Do you? 😄🔥",
        "Code": "def greet(name):\n    return f\"Hello, {name}!\"",
        "Mixed": "价格是€12.50 — great deal, right?",
    }

    default = demo_texts["English sentence"]
    text = st.text_area("Input text", value=default, height=160)

    uploaded = st.file_uploader("…or upload a UTF-8 text file", type=["txt"])
    if uploaded:
        try:
            text = uploaded.read().decode("utf-8")
        except Exception:
            st.warning("Could not decode file as UTF-8; using original input.")

    st.subheader("Tokenizer A")
    tok_a_type = st.selectbox(
        "Type", ["tiktoken (OpenAI)", "Hugging Face (transformers)"], index=0
    )

    tiktoken_choices = [
        # Common encodings
        "cl100k_base",   # GPT-4/3.5 era
        "o200k_base",    # GPT-4o family
        "p50k_base",
        "r50k_base",
    ]
    hf_default = "gpt2"  # small & universally available
    tok_a_name = ""
    if tok_a_type.startswith("tiktoken"):
        tok_a_name = st.selectbox("Encoding (tiktoken)", tiktoken_choices, index=0)
    else:
        tok_a_name = st.text_input("HF tokenizer name", value=hf_default, help="Any tokenizer name from Hugging Face Hub (e.g., 'gpt2', 'bert-base-uncased', 'meta-llama/Meta-Llama-3-8B').")

    st.subheader("Tokenizer B (optional)")
    compare = st.checkbox("Enable side-by-side comparison", value=False)

    tok_b_type, tok_b_name = None, None
    if compare:
        tok_b_type = st.selectbox(
            "Type (B)", ["tiktoken (OpenAI)", "Hugging Face (transformers)"], index=1
        )
        if tok_b_type.startswith("tiktoken"):
            tok_b_name = st.selectbox("Encoding (tiktoken) (B)", tiktoken_choices, index=1, key="tok_b_tick")
        else:
            tok_b_name = st.text_input("HF tokenizer name (B)", value="gpt2", key="tok_b_hf")

    st.divider()
    st.markdown("**Install tips:**")
    st.code("pip install streamlit\npip install tiktoken transformers", language="bash")
    st.caption("Hugging Face tokenizers download once and cache locally.")

# -------- Tokenize & Render --------
def run_tokenizer(kind: str, name: str, text: str) -> Tuple[List[int], List[str], str]:
    try:
        if kind.startswith("tiktoken"):
            if tiktoken is None:
                raise RuntimeError("tiktoken is not installed")
            ids, toks = tokenize_with_tiktoken(text, name)
            used = f"tiktoken • {name}"
        else:
            ids, toks = tokenize_with_hf(text, name)
            used = f"transformers • {name}"
        return ids, toks, used
    except Exception as e:
        return [], [], f"Error: {e}"

col_a, col_b = st.columns(2) if compare else (st.container(), None)

with col_a:
    st.subheader("Tokenizer A")
    ids_a, toks_a, used_a = run_tokenizer(tok_a_type, tok_a_name, text)
    if used_a.startswith("Error"):
        st.error(used_a)
    else:
        st.caption(used_a)
        st.markdown("**Token boundaries**")
        st.code(pretty_boundary(toks_a))
        st.markdown(f"**Token count:** {len(ids_a)}")
        with st.expander("Per-token details"):
            st.dataframe(
                {
                    "index": list(range(len(toks_a))),
                    "token": toks_a,
                    "token_id": ids_a,
                    "utf8_bytes(hex)": [token_bytes(t) for t in toks_a],
                    "len(chars)": [len(t) for t in toks_a],
                },
                use_container_width=True,
            )

with col_b if compare else st.container():
    if compare:
        st.subheader("Tokenizer B")
        ids_b, toks_b, used_b = run_tokenizer(tok_b_type, tok_b_name, text)
        if used_b.startswith("Error"):
            st.error(used_b)
        else:
            st.caption(used_b)
            st.markdown("**Token boundaries**")
            st.code(pretty_boundary(toks_b))
            st.markdown(f"**Token count:** {len(ids_b)}")
            with st.expander("Per-token details"):
                st.dataframe(
                    {
                        "index": list(range(len(toks_b))),
                        "token": toks_b,
                        "token_id": ids_b,
                        "utf8_bytes(hex)": [token_bytes(t) for t in toks_b],
                        "len(chars)": [len(t) for t in toks_b],
                    },
                    use_container_width=True,
                )

if compare and ids_a and 'ids_b' in locals():
    st.divider()
    st.subheader("🔍 Differences")
    diff_cols = st.columns(3)
    with diff_cols[0]:
        st.metric("A tokens", len(ids_a))
    with diff_cols[1]:
        st.metric("B tokens", len(ids_b))
    with diff_cols[2]:
        ratio = (len(ids_b) / len(ids_a)) if len(ids_a) else float("nan")
        st.metric("B/A token ratio", f"{ratio:.2f}" if ratio == ratio else "—")

    with st.expander("Aligned view (first 100 tokens each)"):
        max_show = 100
        rows = []
        for i in range(max(len(ids_a), len(ids_b))):
            a_tok = toks_a[i] if i < len(toks_a) else ""
            b_tok = toks_b[i] if i < len(toks_b) else ""
            a_id = ids_a[i] if i < len(ids_a) else ""
            b_id = ids_b[i] if i < len(ids_b) else ""
            rows.append({
                "pos": i,
                "A_token": a_tok.replace("\n", "\\n"),
                "A_id": a_id,
                "B_token": b_tok.replace("\n", "\\n"),
                "B_id": b_id,
                "same_token_str": a_tok == b_tok,
                "same_id": a_id == b_id,
            })
            if i >= max_show - 1:
                break
        st.dataframe(rows, use_container_width=True)

st.divider()
st.markdown(
    """
**Notes**

- *tiktoken* encodings (e.g., `cl100k_base`, `o200k_base`) are used by OpenAI models and are byte-pair-encoding (BPE)-style over UTF-8 bytes.  
- *transformers* tokenizers vary (BPE, WordPiece, SentencePiece). Try models like `gpt2`, `bert-base-uncased`, `roberta-base`, or a LLaMA tokenizer repo.  
- Token counts differ by tokenizer—this demo helps you see where and why.
"""
)
