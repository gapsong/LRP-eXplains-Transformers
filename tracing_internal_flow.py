import torch
from transformers import AutoTokenizer
from lxt.models.llama import LlamaForCausalLM, attnlrp
import matplotlib.pyplot as plt
import numpy as np


def save_heatmap(values, tokens, figsize, title, save_path):
    fig, ax = plt.subplots(figsize=figsize)

    abs_max = abs(values).max()
    im = ax.imshow(values, cmap="bwr", vmin=-abs_max, vmax=abs_max)

    layers = np.arange(values.shape[-1])

    print(type(values))
    print(np.array(values))  # Check if it converts properly

    ax.set_xticks(np.arange(len(layers)))
    ax.set_yticks(np.arange(len(tokens)))

    ax.set_xticklabels(layers)
    ax.set_yticklabels(tokens)

    plt.title(title)
    plt.xlabel("Layers")
    plt.ylabel("Tokens")
    plt.colorbar(im)

    plt.show()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")


def hidden_relevance_hook(module, input, output):

    if isinstance(output, tuple):
        output = output[0]

    relevance = output.detach().cpu().float().numpy()  # Convert to NumPy array
    module.hidden_relevance = relevance

    for name, param in model.named_modules():
        if param is module:
            if name not in relevance_state.keys():
                relevance_state[name] = (
                    module  # {"relevance": relevance, "min": relevance[0].min(), "max": relevance[0].max(), "sum": relevance[0].sum()}
                )
            break


def apply_hooks_to_leaf_modules(module):

    if len(list(module.children())) == 0:
        module.register_full_backward_hook(hidden_relevance_hook)

    else:
        for child in module.children():
            apply_hooks_to_leaf_modules(child)


relevance_state = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cache_dir = "/home/nico/projects/llm_x_subspaces/models"
read_token_huggingface = "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"

if not torch.cuda.is_available():
    cache_dir = "/Users/harder/PycharmProjects/llm_x_subspaces/models"

model = LlamaForCausalLM.from_pretrained(
    "Felladrin/Llama-160M-Chat-v1",
    torch_dtype=torch.bfloat16,
    device_map=device,
)

tokenizer = AutoTokenizer.from_pretrained("Felladrin/Llama-160M-Chat-v1")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.eval()
attnlrp.register(model)

model_dict = model.state_dict()

apply_hooks_to_leaf_modules(model)

for layer in model.model.layers:
    layer.register_full_backward_hook(hidden_relevance_hook)

prompt_response = f"<s>The richest person in the world is"

input_ids = tokenizer(
    prompt_response, padding=True, truncation=True,  return_tensors="pt", add_special_tokens=False
).input_ids.to(model.device)
input_embeds = model.get_input_embeddings()(input_ids)

output_logits = model(
    inputs_embeds=input_embeds.requires_grad_(), use_cache=False
).logits
max_logits, max_indices = torch.max(output_logits[:, -1, :], dim=-1)
max_logits.backward(max_logits)

relevance_trace = []
relevance_dict = {}
for layer in model.model.layers:
    relevance = layer.hidden_relevance[0].sum(-1)
    relevance = torch.tensor(relevance)
    relevance = relevance / relevance.abs().max()
    relevance_trace.append(relevance)
    relevance_dict[layer] = relevance

relevance_trace = torch.stack(relevance_trace)
float_relevance_trace = relevance_trace.float().numpy().T
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

save_heatmap(
    float_relevance_trace,
    tokens,
    (10, 7),
    f"Latent Relevance Trace (Normalized)",
    f"latent_rel_trace.png",
)

num_tokens = 15
predicted_tokens = []

for _ in range(num_tokens):
    output_logits = model(
        inputs_embeds=input_embeds.requires_grad_(), use_cache=False
    ).logits
    max_logits, max_indices = torch.max(output_logits[:, -1, :], dim=-1)
    predicted_tokens.append(max_indices.item())
    input_ids = torch.cat([input_ids, max_indices.unsqueeze(0)], dim=1)
    input_embeds = model.get_input_embeddings()(input_ids)

predicted_text = tokenizer.decode(predicted_tokens)
print(f"\nPrompt:", prompt_response)
print(f"\nPrediction:", predicted_text)
