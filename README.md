# 🔤 LLM Tokenization Explorer

A **Streamlit** app for exploring how different Large Language Model (LLM) tokenizers split text.
Compare **OpenAI's tiktoken** encodings with **Hugging Face Transformers** tokenizers, see token boundaries, UTF-8 byte values, token IDs, and even compare two tokenizers side-by-side.

## ✨ Features

* **Supports multiple tokenizer backends**

  * OpenAI’s [`tiktoken`](https://github.com/openai/tiktoken)
  * Hugging Face [`transformers`](https://github.com/huggingface/transformers)
* **Visual token boundaries** (`│token│token│`)
* **Token metadata**

  * Token ID
  * UTF-8 byte representation
  * Character length
* **Side-by-side comparison**

  * Compare token counts
  * View aligned tokens
  * Spot differences easily
* **Sample multilingual texts & emoji handling**
* **File upload support** (UTF-8 `.txt` files)

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-tokenization-explorer.git
cd llm-tokenization-explorer

# Install dependencies
pip install streamlit tiktoken transformers
```

## 🖥 Usage

Run the app with:

```bash
streamlit run app.py
```

Then open the provided **local URL** (usually `http://localhost:8501`) in your browser.

## 📋 How It Works

1. **Choose Tokenizer A**
   Select either a `tiktoken` encoding (e.g., `cl100k_base`, `o200k_base`) or a Hugging Face tokenizer (e.g., `gpt2`, `bert-base-uncased`).

2. **(Optional) Enable Comparison**
   Add Tokenizer B for side-by-side analysis.

3. **Input Text**

   * Use the provided demo examples
   * Paste your own text
   * Or upload a `.txt` file

4. **View Results**

   * Token boundaries
   * Detailed token table (IDs, bytes, char length)
   * Comparison metrics and aligned token view

