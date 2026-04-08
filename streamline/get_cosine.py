import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoConfig


def generate_unique_index(start, end, length):
    return torch.randperm(end - start)[:length].tolist()


def get_cos_similar_matrix(v1, v2, device):
    v1 = v1.to(device)
    v2 = v2.to(device)
    num = torch.mm(v1, v2.t())
    denom = torch.norm(v1, dim=1).reshape(-1, 1) * torch.norm(v2, dim=1)
    res = num / denom
    res[torch.isinf(res)] = 0
    res = 0.5 + 0.5 * res
    res = res.cpu()

    del v1, v2, num, denom
    return res


def average_similarity(layer_cosine_similarity):
    return torch.tensor(layer_cosine_similarity).mean().item()


@torch.no_grad()
def get_cosine_similarity(model, loader, device, layer_intervals, num_layer):
    hidden_states_list = []

    for batch in tqdm(loader, desc="Collecting hidden states"):
        hidden_states = model(batch['input_ids'], output_hidden_states=True).hidden_states
        hidden_states = [h.cpu() for h in hidden_states]
        hidden_states_list.append(hidden_states)

        del batch

    cosine_similarity = [[] for _ in range(num_layer - layer_intervals + 1)]

    for i in range(len(hidden_states_list)):
        for j in range(num_layer - layer_intervals + 1):
            cosine = get_cos_similar_matrix(
                hidden_states_list[i][j][0],
                hidden_states_list[i][j + layer_intervals][0],
                device
            )
            similarity = torch.trace(cosine.to(torch.float)) / cosine.size(0)
            cosine_similarity[j].append(similarity.item())
            del cosine

    print('Calculating cosine similarity...')
    similarities = [average_similarity(layer_sim) for layer_sim in cosine_similarity]
    similarities_tensor = torch.tensor(similarities)

    best_layer = torch.argmax(similarities_tensor).item()
    best_cosine = similarities[best_layer]

    for i, sim in enumerate(similarities):
        print(f'The cosine similarity between hidden_states {i} and hidden_states {i + layer_intervals} is {sim:.4f}')

    print(
        f'The highest cosine similarity comes from hidden_states {best_layer} and hidden_states {best_layer + layer_intervals}, with a value of {best_cosine:.4f}')

    model.cpu()
    del hidden_states_list, model
    torch.cuda.empty_cache()

    return best_layer


if __name__ == '__main__':
    config_name = 'Qwen/Qwen3-8B-Base'
    config = AutoConfig.from_pretrained(config_name)

    num_samples = 50
    layer_intervals = 22
    num_hidden_layers = config.num_hidden_layers
    device = torch.device('cuda:0')

    model = AutoModelForCausalLM.from_pretrained(config_name, device_map=device)
    model.eval()

    ds = load_from_disk('data/tokenized/finewebedu_qwen3_train')
    ds = ds.with_format("torch", device=device).shuffle(42).select(range(num_samples))
    dl = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False)

    get_cosine_similarity(model, dl, device, layer_intervals, num_hidden_layers)
