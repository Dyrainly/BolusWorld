'Copyright 2025 NUS Show Lab.\n\nLicensed under the Apache License, Version 2.0 (the "License");\nyou may not use this file except in compliance with the License.\nYou may obtain a copy of the License at\n\n    http://www.apache.org/licenses/LICENSE-2.0\n\nUnless required by applicable law or agreed to in writing, software\ndistributed under the License is distributed on an "AS IS" BASIS,\nWITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\nSee the License for the specific language governing permissions and\nlimitations under the License.\n\nUpstream source adapted for the BolusWorld release. Source comments were removed. See the accompanying Apache License and third-party notices.'

import numpy as np
import torch
from torch.nn.attention.flex_attention import BlockMask
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

if torch.cuda.is_available():
    flex_attention = torch.compile(flex_attention)


def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def full(b, h, q_idx, kv_idx):
    return q_idx >= 0


def modality(offset, length):
    def mask_fn(b, h, q_idx, kv_idx):
        return (q_idx >= offset) & (kv_idx < (offset + length))

    return mask_fn



def omni_attn_mask(modalities):
    modalities = modalities.long()

    def mask_mod(b, h, q_idx, kv_idx):
        mask = causal(b, h, q_idx, kv_idx)

        modality_batch = modalities[b]

        for offset, length in modality_batch:
            mask = mask | modality(offset, length)(b, h, q_idx, kv_idx)

        return mask

    return mask_mod


def omni_attn_mask_naive(B, LEN, modalities, device, inverted=True):
    attention_mask = torch.tril(torch.ones((B, 1, LEN, LEN), dtype=torch.long)).to(device)
    for b in range(B):
        modality_batch = modalities[b]
        for offset, length in modality_batch:
            attention_mask[b, :, offset:offset + length, offset:offset + length] = 1

    if inverted:
        inverted_attention_mask = 1 - attention_mask
        inverted_attention_mask = inverted_attention_mask.masked_fill(
            inverted_attention_mask.to(torch.bool), torch.iinfo(torch.long).min
        )
        return inverted_attention_mask
    else:
        return attention_mask


def full_attn_mask_naive(B, LEN, device, inverted=True):
    attention_mask = torch.ones((B, 1, LEN, LEN), dtype=torch.long).to(device)
    if inverted:
        inverted_attention_mask = 1 - attention_mask
        inverted_attention_mask = inverted_attention_mask.masked_fill(
            inverted_attention_mask.to(torch.bool), torch.iinfo(torch.long).min
        )
        return inverted_attention_mask
    else:
        return attention_mask

def causal_attn_mask_naive(B, LEN, device, inverted=True):
    attention_mask = torch.tril(torch.ones((B, 1, LEN, LEN), dtype=torch.long)).to(device)
    if inverted:
        inverted_attention_mask = 1 - attention_mask
        inverted_attention_mask = inverted_attention_mask.masked_fill(
            inverted_attention_mask.to(torch.bool), torch.iinfo(torch.long).min
        )
        return inverted_attention_mask
    else:
        return attention_mask

if __name__ == '__main__':
    device = 'cuda:0'
















    seq_len = 20
    modality_positions = torch.from_numpy(np.array(
        [[(3, 8), (0, 0), (0, 0)], [(0, 5), (10, 15), (0, 0)]])).to(device)
    omni_mask = omni_attn_mask_naive(2, seq_len, modality_positions, device, inverted=False)
    print(omni_mask)
