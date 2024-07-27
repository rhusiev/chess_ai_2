import numpy as np
import torch


def numpy_to_tensor(numpy_array):
    return torch.from_numpy(numpy_array).float()


if __name__ == "__main__":
    DATA_DIR = "data/20.11/"
    paths = [
        f"{DATA_DIR}moves_tensors_1600-2000.npy",
        f"{DATA_DIR}states_consts_tensors_1600-2000.npy",
        f"{DATA_DIR}states_tensors_1600-2000.npy",
    ]
    for path in paths:
        numpy_array = np.load(path)
        tensor = numpy_to_tensor(numpy_array)
        del numpy_array
        torch.save(tensor, path.replace(".npy", ".pt"))
        # clear memory
        del tensor
        torch.cuda.empty_cache()
        print(f"Converted {path} to tensor.")
