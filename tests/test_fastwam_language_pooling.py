import torch

from fastwam.models.wan22.fastwam import FastWAM


class _Tokenizer:
    def __call__(self, prompt, *, return_mask, add_special_tokens):
        assert return_mask and add_special_tokens
        batch_size = len(prompt)
        ids = torch.tensor([[1, 2, 0], [3, 0, 0]], dtype=torch.long)[:batch_size]
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.long)[:batch_size]
        return ids, mask


class _TextEncoder(torch.nn.Module):
    def forward(self, ids, mask):
        return torch.stack((ids.float(), ids.float() * 10.0), dim=-1)


def test_prompt_pooling_uses_true_mask_without_changing_released_attention_mask():
    model = FastWAM.__new__(FastWAM)
    torch.nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.tokenizer = _Tokenizer()
    model.text_encoder = _TextEncoder()

    pooled = model.encode_prompt_pooled(["first", "second"])
    torch.testing.assert_close(
        pooled,
        torch.tensor([[1.5, 15.0], [3.0, 30.0]], dtype=torch.float32),
    )
    _, released_mask = model.encode_prompt(["first", "second"])
    assert torch.equal(released_mask, torch.ones_like(released_mask))
