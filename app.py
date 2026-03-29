import argparse
import csv
import json
import time
from pathlib import Path

from agents.intent_classifier import classify
from generation.response_generator import generate
from retrieval.vector_store import load_index, retrieve
from agents.guardrails import check_input, check_output

TIMINGS_FILE = Path("timings.csv")
TIMING_FIELDS = [
    "turn_id",
    "classify",
    "retrieve_embed",
    "retrieve_faiss",
    "retrieve_rerank",
    "retrieve_total",
    "generate",
    "total",
]


def init_timings_csv():
    with TIMINGS_FILE.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TIMING_FIELDS)
        writer.writeheader()


def append_timing_row(row):
    with TIMINGS_FILE.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TIMING_FIELDS)
        writer.writerow(row)


def print_timing_table(row):
    headers = [
        "turn_id",
        "classify",
        "ret_embed",
        "ret_faiss",
        "ret_rerank",
        "ret_total",
        "generate",
        "total",
    ]
    values = [
        str(row["turn_id"]),
        f"{row['classify']:.3f}",
        f"{row['retrieve_embed']:.3f}",
        f"{row['retrieve_faiss']:.3f}",
        f"{row['retrieve_rerank']:.3f}",
        f"{row['retrieve_total']:.3f}",
        f"{row['generate']:.3f}",
        f"{row['total']:.3f}",
    ]
    widths = [max(len(h), len(v)) for h, v in zip(headers, values)]
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    value_line = " | ".join(v.ljust(w) for v, w in zip(values, widths))
    print("\n[timing]")
    print(header_line)
    print(value_line)


parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true", help="Enable latency profiling output")
args = parser.parse_args()
debug_timing = args.debug

if debug_timing:
    init_timings_csv()

with open("data/users.json") as f:
    users = json.load(f)["users"]

print("Available users:")
for u in users:
    print(f"  {u['id']}")

user_id = input("Select user (type the id exactly): ").strip()

valid_ids = [u["id"] for u in users]
if user_id not in valid_ids:
    print(f"Invalid id. Choose from: {valid_ids}")
    raise SystemExit(1)

index, meta = load_index(f"data/faiss_store/{user_id}")

from retrieval.vector_store import embedder, reranker
embedder.encode(["warmup"], convert_to_numpy=True)
reranker.predict([("warmup", "warmup")])
print(f"Loaded {user_id}. Models warmed up. Start chatting!\n")


turn_id = 0
while True:
    query = input("User: ")
    turn_id += 1
    t_turn_start = time.perf_counter()

    t0 = time.perf_counter()
    intent_info = classify(query)
    t_classify = time.perf_counter() - t0

    t0 = time.perf_counter()
    if debug_timing:
        memories, retrieve_timing = retrieve(
            query,
            index,
            meta,
            top_k=5,
            rerank_k=3,
            bucket_filter=intent_info["bucket"],
            debug=True,
        )
    else:
        memories = retrieve(
            query,
            index,
            meta,
            top_k=5,
            rerank_k=3,
            bucket_filter=intent_info["bucket"],
        )
        retrieve_timing = {
            "retrieve_embed": 0.0,
            "retrieve_faiss": 0.0,
            "retrieve_rerank": 0.0,
            "retrieve_total": 0.0,
        }
    t_retrieve_total = time.perf_counter() - t0
    if debug_timing:
        retrieve_timing["retrieve_total"] = t_retrieve_total

    t0 = time.perf_counter()
    response = generate(query, memories, user_id)
    t_generate = time.perf_counter() - t0

    t_total = time.perf_counter() - t_turn_start
    print("AAC Bot:", response)

    if debug_timing:
        row = {
            "turn_id": turn_id,
            "classify": round(t_classify * 1000, 3),
            "retrieve_embed": round(retrieve_timing["retrieve_embed"] * 1000, 3),
            "retrieve_faiss": round(retrieve_timing["retrieve_faiss"] * 1000, 3),
            "retrieve_rerank": round(retrieve_timing["retrieve_rerank"] * 1000, 3),
            "retrieve_total": round(retrieve_timing["retrieve_total"] * 1000, 3),
            "generate": round(t_generate * 1000, 3),
            "total": round(t_total * 1000, 3),
        }
        append_timing_row(row)
        print_timing_table(row)