import torch


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

def export_gpt(model: str, output: str):
    pass

if __name__ == "__main__":
    weights = torch.load("./models/pytorch_model.bin", map_location="cpu")

    summarize_weights(weights)
    export_gpt(weights, "c_model.bin")