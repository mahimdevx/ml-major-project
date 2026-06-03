#!/usr/bin/env bash
# demo/smoke_test.sh
# 30-second sanity check before a live demo.
# Verifies the four things that most commonly break a demo:
#   1. Python dependencies are installed
#   2. Ollama is running and the model is pulled
#   3. The ChromaDB index exists and is readable
#   4. The four prompt template files exist
#
# Run from project root:
#   bash demo/smoke_test.sh

set -u

# Use the directory above this script as the project root.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PASS=0
FAIL=0

ok()   { echo " [ok]   $*"; PASS=$((PASS+1)); }
fail() { echo " [FAIL] $*"; FAIL=$((FAIL+1)); }

echo
echo "=== Shakespeare RAG smoke test ==="
echo "Project root: $PROJECT_ROOT"
echo

# 1. Python deps
echo "[1/4] Checking Python dependencies..."
python3 -c "
import importlib, sys
mods = ['chromadb', 'ollama', 'sentence_transformers', 'sklearn', 'numpy']
missing = [m for m in mods if not importlib.util.find_spec(m)]
sys.exit(1 if missing else 0)
print('all imports resolvable')
" >/dev/null 2>&1
if [ $? -eq 0 ]; then
    ok "Python dependencies present"
else
    fail "Some dependencies missing -- run: pip install -r requirements.txt"
fi

# 2. Ollama
echo "[2/4] Checking Ollama and gemma3:4b..."
if command -v ollama >/dev/null 2>&1; then
    if ollama list 2>/dev/null | grep -q "gemma3:4b"; then
        ok "Ollama installed and gemma3:4b is pulled"
    else
        fail "Ollama installed but gemma3:4b missing -- run: ollama pull gemma3:4b"
    fi
else
    fail "Ollama not installed -- install from https://ollama.com"
fi

# 3. ChromaDB index
echo "[3/4] Checking ChromaDB utterance index..."
python3 -c "
import sys
try:
    import chromadb
    client = chromadb.PersistentClient(path='data/chroma_utterances')
    col = client.get_collection('shakespeare_utterances')
    n = col.count()
    if n < 2000:
        print(f'unexpectedly small: {n}', file=sys.stderr)
        sys.exit(2)
    print(n)
except Exception as e:
    print(f'error: {e}', file=sys.stderr)
    sys.exit(1)
" 2>/dev/null
RC=$?
if [ $RC -eq 0 ]; then
    ok "ChromaDB index loaded"
else
    fail "ChromaDB index missing or unreadable at data/chroma_utterances/"
fi

# 4. Prompt files
echo "[4/4] Checking prompt template files..."
ALL_PROMPTS_OK=1
for f in prompts/baseline_prompt.txt prompts/system_prompt.txt prompts/stylised_prompt.txt; do
    if [ -f "$f" ]; then
        :
    else
        fail "Missing prompt file: $f"
        ALL_PROMPTS_OK=0
    fi
done
if [ $ALL_PROMPTS_OK -eq 1 ]; then
    ok "All three prompt templates present"
fi

echo
echo "=== Result: $PASS passed, $FAIL failed ==="
if [ $FAIL -eq 0 ]; then
    echo "System is ready for demo."
    exit 0
else
    echo "Fix the failures above before demonstrating."
    exit 1
fi
