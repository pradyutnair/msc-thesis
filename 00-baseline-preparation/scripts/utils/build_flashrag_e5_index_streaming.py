import argparse
import json
import os

import faiss
from flashrag.retriever.encoder import Encoder


def iter_corpus_texts(corpus_path):
    with open(corpus_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "contents" in item:
                yield item["contents"]
            elif "text" in item:
                yield item["text"]


def build_index(args):
    os.makedirs(args.save_dir, exist_ok=True)

    encoder = Encoder(
        model_name=args.retrieval_method,
        model_path=args.model_path,
        pooling_method=args.pooling_method,
        max_length=args.max_length,
        use_fp16=args.use_fp16,
        instruction=args.instruction,
    )

    index = None
    total_docs = 0
    buffer = []

    for text in iter_corpus_texts(args.corpus_path):
        buffer.append(text)
        if len(buffer) >= args.chunk_size:
            embeddings = encoder.encode(buffer, batch_size=args.batch_size, is_query=False)
            if index is None:
                dim = embeddings.shape[-1]
                index = faiss.index_factory(dim, args.faiss_type, faiss.METRIC_INNER_PRODUCT)
                if not index.is_trained:
                    index.train(embeddings)
            index.add(embeddings)
            total_docs += len(buffer)
            buffer.clear()
            print(f"Indexed {total_docs} documents...")

    if buffer:
        embeddings = encoder.encode(buffer, batch_size=args.batch_size, is_query=False)
        if index is None:
            dim = embeddings.shape[-1]
            index = faiss.index_factory(dim, args.faiss_type, faiss.METRIC_INNER_PRODUCT)
            if not index.is_trained:
                index.train(embeddings)
        index.add(embeddings)
        total_docs += len(buffer)
        print(f"Indexed {total_docs} documents...")

    if index is None:
        raise RuntimeError("No documents were indexed. Check corpus path and format.")

    index_save_path = os.path.join(args.save_dir, f"{args.retrieval_method}_{args.faiss_type}.index")
    faiss.write_index(index, index_save_path)
    print(f"Saved index to {index_save_path}")


def main():
    parser = argparse.ArgumentParser(description="Streaming FAISS index builder.")
    parser.add_argument("--retrieval_method", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--corpus_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--faiss_type", type=str, default="Flat")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--chunk_size", type=int, default=10000)
    parser.add_argument("--use_fp16", action="store_true", default=False)
    parser.add_argument("--pooling_method", type=str, default=None)
    parser.add_argument("--instruction", type=str, default=None)
    args = parser.parse_args()

    build_index(args)


if __name__ == "__main__":
    main()
