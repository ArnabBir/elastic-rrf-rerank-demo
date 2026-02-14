import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import read_jsonl, get_data_dir

def main():
    data_dir = get_data_dir()
    docs = read_jsonl(data_dir / "documents.jsonl")
    queries = read_jsonl(data_dir / "queries.jsonl")
    print(f"Loaded {len(docs)} documents")
    print(f"Loaded {len(queries)} queries")
    print("OK")

if __name__ == "__main__":
    main()
