import sys
from pathlib import Path

# Garante que o diretório raiz (onde está src/) esteja no PYTHONPATH ao rodar pytest.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
