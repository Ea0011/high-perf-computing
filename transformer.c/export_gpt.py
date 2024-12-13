import torch
import numpy as np


def summarize_weights(weights: dict):
    for key in weights.keys():
        print(key, weights[key].shape)

    # inspect biases
    print("Biases:\n")
    for key in weights.keys():
        if "bias" in key:
            print(key, weights[key].shape)

    # Attention biases
    # These are the biases that are added to the attention scores before softmax for masking
    print("Attention biases:\n")
    for key in weights.keys():
        if "attn.bias" in key:
            print(key, weights[key].shape, weights[key])

def export_gpt(
    model: dict,
    output: str
):
    attn_weights_to_break = ["c_attn.weight", "c_attn.bias"] # Need to break these down to (head, d_model, head_dim)
    keys_to_traspose = ['c_attn.weight', 'c_proj.weight', 'c_fc.weight', 'c_proj.weight']
    
    num_exported_params = 0
    with open(output, "wb") as f:
        for key in model.keys():
            if key.endswith(".attn.bias") or key.endswith(".attn.masked_bias"):
                continue
            
            if any(key.endswith(k) for k in keys_to_traspose):
                print(f"Transposing {key} with shape {model[key].shape}")
                model[key] = model[key].T

            if any(key.endswith(k) for k in attn_weights_to_break):
                # Break down the attention weights into (head, d_model, head_dim)
                # This is necessary because the weights are stored in the format (3 * d_model, d_model)
                # where the first d_model is for the query, the second for the key, and the third for the value
                if len(model[key].shape) == 2:
                    d_model = model[key].shape[1]
                    model[key] = model[key].view(3, d_model, d_model)
                elif len(model[key].shape) == 1:
                    d_model = model[key].shape[0]
                    model[key] = model[key].view(3, d_model // 3)

            print(f"Writing {key} with shape {model[key].shape} at position {f.tell()}: Size in float32: {model[key].numpy().size * 4}")
            f.write(model[key].numpy().astype(np.float32).tobytes())
            num_exported_params += model[key].numpy().size

        print("Total size in bytes:", f.tell())
        print("Total size in MB:", f.tell() / 1024 / 1024)

        print("Total number of EXPORTED parameters:", num_exported_params)


if __name__ == "__main__":
    weights = torch.load("./models/pytorch_model.bin", map_location="cpu")

    # summarize_weights(weights)
    export_gpt(weights, "./models/c_model.bin")