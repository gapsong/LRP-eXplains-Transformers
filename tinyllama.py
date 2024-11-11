import torch
import numpy as np
from transformers import AutoTokenizer
from lxt.models.llama import LlamaForCausalLM, attnlrp
from lxt.utils import pdf_heatmap, clean_tokens
from collections import OrderedDict

model = LlamaForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    trust_remote_code=True,
    output_attentions=True,
    output_hidden_states=True,
)

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

model.gradient_checkpointing_enable()
# apply AttnLRP rules
attnlrp.register(model)


# Initialize relevance_state to store relevances
relevance_state = OrderedDict()


def hidden_relevance_hook(module, input, output):
    if isinstance(output, tuple):
        output = output[0]
    relevance = output.detach().cpu().float().numpy()  # Convert to NumPy array
    for name, param in model.named_modules():
        if param is module:
            relevance_state[name] = relevance
            print(f"Relevance saved for {name}: shape = {relevance.shape}")
            break


# Apply the hook only to leaf modules
def apply_hooks_to_leaf_modules(module):
    if len(list(module.children())) == 0:
        module.register_full_backward_hook(hidden_relevance_hook)
        print(f"Hook registered on leaf module: {module}")
    else:
        for child in module.children():
            apply_hooks_to_leaf_modules(child)


# Apply hooks to all leaf modules in the model
apply_hooks_to_leaf_modules(model.model)

# After running a forward and backward pass, you can inspect relevance_state
# This dictionary will contain the relevances with keys matching model.state_dict() structure
print(relevance_state)

print(model)  # Check that layers are correctly modified for relevance tracking

prompt = """\
Context: Mount Everest attracts many climbers, including highly experienced mountaineers. There are two main climbing routes, one approaching the summit from the southeast in Nepal (known as the standard route) and the other from the north in Tibet. While not posing substantial technical climbing challenges on the standard route, Everest presents dangers such as altitude sickness, weather, and wind, as well as hazards from avalanches and the Khumbu Icefall. As of November 2022, 310 people have died on Everest. Over 200 bodies remain on the mountain and have not been removed due to the dangerous conditions. The first recorded efforts to reach Everest's summit were made by British mountaineers. As Nepal did not allow foreigners to enter the country at the time, the British made several attempts on the north ridge route from the Tibetan side. After the first reconnaissance expedition by the British in 1921 reached 7,000 m (22,970 ft) on the North Col, the 1922 expedition pushed the north ridge route up to 8,320 m (27,300 ft), marking the first time a human had climbed above 8,000 m (26,247 ft). The 1924 expedition resulted in one of the greatest mysteries on Everest to this day: George Mallory and Andrew Irvine made a final summit attempt on 8 June but never returned, sparking debate as to whether they were the first to reach the top. Tenzing Norgay and Edmund Hillary made the first documented ascent of Everest in 1953, using the southeast ridge route. Norgay had reached 8,595 m (28,199 ft) the previous year as a member of the 1952 Swiss expedition. The Chinese mountaineering team of Wang Fuzhou, Gonpo, and Qu Yinhua made the first reported ascent of the peak from the north ridge on 25 May 1960. \
Question: How high did they climb in 1922? According to the text, the 1922 expedition reached 8,"""

input_ids = tokenizer(
    prompt, return_tensors="pt", add_special_tokens=True
).input_ids.to(model.device)
input_embeds = model.get_input_embeddings()(
    input_ids
).requires_grad_()  # uses the input_ids of the tokens and creates embeddings
print(input_embeds.requires_grad)  # Should be True

output_logits = model(
    inputs_embeds=input_embeds.requires_grad_(), use_cache=False
).logits

# The winning token gets chosen. Then we use the winning token to backtrack the gradients of the model
max_logits, max_indices = torch.max(output_logits[0, -1, :], dim=-1)
max_logits.backward(max_logits)

relevance = input_embeds.grad.float().sum(-1).cpu()[0]
relevance_sum = sum(relevance)
# normalize relevance between [-1, 1] for plotting
relevance = relevance / relevance.abs().max()

# remove '_' characters from token strings
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
tokens = clean_tokens(tokens)

pdf_heatmap(tokens, relevance, path="heatmap.pdf", backend="xelatex")

# Optionally, save as a compressed .npz file (all arrays in one file)
np.savez_compressed("relevance_state.npz", **relevance_state)
print("All relevances saved to relevance_state.npz")
