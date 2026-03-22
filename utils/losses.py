from __future__ import annotations

import torch
import torch.nn.functional as F


def _prepare_labels(
    similarity: torch.Tensor,
    crop_labels: torch.Tensor | list[int],
    disease_labels: torch.Tensor | list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    crop_labels = torch.as_tensor(crop_labels, device=similarity.device, dtype=torch.long)
    disease_labels = torch.as_tensor(disease_labels, device=similarity.device, dtype=torch.long)
    return crop_labels, disease_labels


def _validate_inputs(
    similarity: torch.Tensor,
    crop_labels: torch.Tensor,
    disease_labels: torch.Tensor,
) -> None:
    if similarity.ndim != 2:
        raise ValueError(f"similarity must be 2D [B, B], got shape {tuple(similarity.shape)}")
    if similarity.shape[0] != similarity.shape[1]:
        raise ValueError(f"similarity must be square [B, B], got shape {tuple(similarity.shape)}")
    if crop_labels.ndim != 1 or disease_labels.ndim != 1:
        raise ValueError("crop_labels and disease_labels must be 1D tensors")
    if crop_labels.numel() != similarity.shape[0] or disease_labels.numel() != similarity.shape[0]:
        raise ValueError(
            "Label length must match batch size: "
            f"got crop={crop_labels.numel()}, disease={disease_labels.numel()}, batch={similarity.shape[0]}"
        )


def build_taxonomy_soft_targets(
    similarity: torch.Tensor,
    crop_labels: torch.Tensor | list[int],
    disease_labels: torch.Tensor | list[int],
    epsilon: float = 0.1,
    alpha: float = 0.5,
    beta: float = 0.25,
) -> torch.Tensor:
    crop_labels, disease_labels = _prepare_labels(similarity, crop_labels, disease_labels)
    _validate_inputs(similarity, crop_labels, disease_labels)

    batch_size = similarity.shape[0]
    eps = torch.as_tensor(epsilon, device=similarity.device, dtype=similarity.dtype)
    one_minus_eps = torch.as_tensor(1.0 - epsilon, device=similarity.device, dtype=similarity.dtype)

    eye = torch.eye(batch_size, device=similarity.device, dtype=torch.bool)
    offdiag = ~eye
    same_crop = crop_labels.unsqueeze(1).eq(crop_labels.unsqueeze(0))
    same_disease = disease_labels.unsqueeze(1).eq(disease_labels.unsqueeze(0))

    # Off-diagonal weights in Eq(10): 1 / alpha / beta / 0
    weight_same_pair = offdiag & same_crop & same_disease
    weight_same_disease_diff_crop = offdiag & same_disease & (~same_crop)
    weight_same_crop_diff_disease = offdiag & same_crop & (~same_disease)

    weights = torch.zeros_like(similarity)
    weights = weights + weight_same_pair.to(dtype=similarity.dtype)
    weights = weights + torch.as_tensor(alpha, device=similarity.device, dtype=similarity.dtype) * (
        weight_same_disease_diff_crop.to(dtype=similarity.dtype)
    )
    weights = weights + torch.as_tensor(beta, device=similarity.device, dtype=similarity.dtype) * (
        weight_same_crop_diff_disease.to(dtype=similarity.dtype)
    )

    row_weight_sum = weights.sum(dim=1, keepdim=True)
    has_neighbors = row_weight_sum > 0

    q_matrix = torch.zeros_like(similarity)
    row_weight_sum_safe = row_weight_sum.clamp_min(torch.finfo(similarity.dtype).eps)
    offdiag_probs = eps * weights / row_weight_sum_safe
    q_matrix = torch.where(has_neighbors, offdiag_probs, q_matrix)

    diag_vals = torch.where(
        has_neighbors.squeeze(1),
        torch.full((batch_size,), one_minus_eps.item(), device=similarity.device, dtype=similarity.dtype),
        torch.ones((batch_size,), device=similarity.device, dtype=similarity.dtype),
    )
    q_matrix = q_matrix.masked_fill(eye, 0.0)
    q_matrix = q_matrix + torch.diag(diag_vals)
    return q_matrix


def taxonomy_aware_soft_target_contrastive_loss(
    similarity: torch.Tensor,
    crop_labels: torch.Tensor | list[int],
    disease_labels: torch.Tensor | list[int],
    epsilon: float = 0.1,
    alpha: float = 0.5,
    beta: float = 0.25,
) -> torch.Tensor:
    q_matrix = build_taxonomy_soft_targets(
        similarity=similarity,
        crop_labels=crop_labels,
        disease_labels=disease_labels,
        epsilon=epsilon,
        alpha=alpha,
        beta=beta,
    )

    loss_i2t = F.cross_entropy(similarity, q_matrix)
    loss_t2i = F.cross_entropy(similarity.T, q_matrix.T)
    return 0.5 * (loss_i2t + loss_t2i)


if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sim = torch.randn(8, 8, device=device, dtype=torch.float32, requires_grad=True)
    crop_labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 4], device=device, dtype=torch.long)
    disease_labels = torch.tensor([0, 1, 0, 1, 2, 2, 3, 5], device=device, dtype=torch.long)

    q = build_taxonomy_soft_targets(sim, crop_labels, disease_labels)
    loss = taxonomy_aware_soft_target_contrastive_loss(sim, crop_labels, disease_labels)
    loss.backward()

    row_sums = q.sum(dim=1)
    print("similarity shape:", tuple(sim.shape))
    print("q_matrix shape:", tuple(q.shape))
    print("similarity device/dtype:", sim.device, sim.dtype)
    print("q_matrix device/dtype:", q.device, q.dtype)
    print("row sums:", row_sums)
    print("loss:", float(loss.item()))
    print("grad finite:", bool(torch.isfinite(sim.grad).all()))
    print("grad norm:", float(sim.grad.norm().item()))

    assert q.shape == (8, 8)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
    assert torch.isfinite(loss)
    assert sim.grad is not None
    assert torch.isfinite(sim.grad).all()

    if torch.cuda.is_available():
        sim_amp = torch.randn(8, 8, device=device, dtype=torch.float32, requires_grad=True)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            loss_amp = taxonomy_aware_soft_target_contrastive_loss(sim_amp, crop_labels, disease_labels)
        loss_amp.backward()
        print("amp loss:", float(loss_amp.item()))
        print("amp grad finite:", bool(torch.isfinite(sim_amp.grad).all()))

    print("Taxonomy-aware loss check passed.")

