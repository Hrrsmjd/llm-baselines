from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset
import os


SPJ_DATA_PATH = os.path.join(os.path.dirname(__file__), "datasets/slimpajama6B/")
SPJ_CHUNK_1_DATA_PATH = os.path.join(SPJ_DATA_PATH, "chunk1")


tknzr = tiktoken.get_encoding("gpt2")


def get_slimpajama_data(num_proc=40):
    if not os.path.exists(os.path.join(SPJ_DATA_PATH, "train.bin")):
        os.makedirs(SPJ_DATA_PATH, exist_ok=True)
        # dataset = load_dataset("DKYoon/SlimPajama-6B") ## NEW
        dataset = load_dataset("DKYoon/SlimPajama-6B", cache_dir=SPJ_DATA_PATH) ## NEW

        split_dataset = dataset["train"].train_test_split(
            test_size=0.0005, seed=2357, shuffle=True
        )
        split_dataset["val"] = split_dataset.pop("test")

        def process(example):
            ids = tknzr.encode_ordinary(
                example["text"]
            )  # encode_ordinary ignores any special tokens
            ids.append(
                tknzr.eot_token
            )  # add the end of text token, e.g. 50256 for gpt2 bpe
            out = {"ids": ids, "len": len(ids)}
            return out

        # tokenize the dataset
        tokenized = split_dataset.map(
            process,
            remove_columns=["text"],
            desc="tokenizing the splits",
            num_proc=num_proc,
        )

        # concatenate all the ids in each dataset into one large file we can use for training
        for split, dset in tokenized.items():
            arr_len = np.sum(dset["len"])
            filename = os.path.join(SPJ_DATA_PATH, f"{split}.bin")
            dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
            arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
            total_batches = min(1024, len(dset))

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
                # Batch together samples for faster write
                batch = dset.shard(
                    num_shards=total_batches, index=batch_idx, contiguous=True
                ).with_format("numpy")
                arr_batch = np.concatenate(batch["ids"])
                # Write into mmap
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()

    train_data = np.memmap(
        os.path.join(SPJ_DATA_PATH, "train.bin"), dtype=np.uint16, mode="r"
    )
    val_data = np.memmap(
        os.path.join(SPJ_DATA_PATH, "val.bin"), dtype=np.uint16, mode="r"
    )

    return {"train": train_data, "val": val_data}


def get_slimpajama_chunk1(num_proc=40):
    if not os.path.exists(os.path.join(SPJ_CHUNK_1_DATA_PATH, "train.bin")):
        # os.makedirs(SPJ_DATA_PATH, exist_ok=True) ## NEW
        # dataset = load_dataset("cerebras/SlimPajama-627B", split="train/chunk1") ## NEW
        os.makedirs(SPJ_CHUNK_1_DATA_PATH, exist_ok=True) ## NEW
        dataset = load_dataset("cerebras/SlimPajama-627B", split="train/chunk1", 
                             cache_dir=SPJ_CHUNK_1_DATA_PATH) ## NEW

        # split_dataset = dataset["train"].train_test_split( ## NEW
        split_dataset = dataset.train_test_split( ## NEW
            test_size=0.0005, seed=2357, shuffle=True
        )
        split_dataset["val"] = split_dataset.pop("test")

        def process(example):
            ids = tknzr.encode_ordinary(
                example["text"]
            )  # encode_ordinary ignores any special tokens
            ids.append(
                tknzr.eot_token
            )  # add the end of text token, e.g. 50256 for gpt2 bpe
            out = {"ids": ids, "len": len(ids)}
            return out

        # tokenize the dataset
        tokenized = split_dataset.map(
            process,
            remove_columns=["text"],
            desc="tokenizing the splits",
            num_proc=num_proc,
        )

        # concatenate all the ids in each dataset into one large file we can use for training
        for split, dset in tokenized.items():
            arr_len = np.sum(dset["len"])
            # filename = os.path.join(SPJ_DATA_PATH, f"{split}.bin") ## NEW
            filename = os.path.join(SPJ_CHUNK_1_DATA_PATH, f"{split}.bin") ## NEW
            dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
            arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
            total_batches = min(1024, len(dset))

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
                # Batch together samples for faster write
                batch = dset.shard(
                    num_shards=total_batches, index=batch_idx, contiguous=True
                ).with_format("numpy")
                arr_batch = np.concatenate(batch["ids"])
                # Write into mmap
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()

    train_data = np.memmap(
        # os.path.join(SPJ_DATA_PATH, "train.bin"), dtype=np.uint16, mode="r" ## NEW
        os.path.join(SPJ_CHUNK_1_DATA_PATH, "train.bin"), dtype=np.uint16, mode="r" ## NEW
    )
    val_data = np.memmap(
        # os.path.join(SPJ_DATA_PATH, "val.bin"), dtype=np.uint16, mode="r" ## NEW
        os.path.join(SPJ_CHUNK_1_DATA_PATH, "val.bin"), dtype=np.uint16, mode="r" ## NEW
    )

    return {"train": train_data, "val": val_data}
