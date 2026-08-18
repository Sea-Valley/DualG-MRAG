from __future__ import annotations

from typing import Any

import math

import torch
from torch_geometric.utils import softmax


def _align_last_dim(tensor: torch.Tensor, target_dim: int) -> torch.Tensor:
    current_dim = int(tensor.shape[-1])
    if current_dim == target_dim:
        return tensor
    if current_dim > target_dim:
        return tensor[..., :target_dim]
    pad_shape = list(tensor.shape[:-1]) + [target_dim - current_dim]
    padding = tensor.new_zeros(pad_shape)
    return torch.cat([tensor, padding], dim=-1)


def normalize_edge_messages_to_flow(
    edge_messages: torch.Tensor,
    target_node_states: torch.Tensor,
    edge_index: torch.Tensor,
    temperature: float = 1.0,
    eps: float = 1e-12,
) -> dict[str, torch.Tensor]:
    if edge_messages.ndim != 3:
        raise ValueError("edge_messages must have shape (batch, num_edges, dim)")
    if target_node_states.ndim != 3:
        raise ValueError(
            "target_node_states must have shape (batch, num_nodes, dim)"
        )

    safe_temperature = max(float(temperature), float(eps))
    dst_index = edge_index[1].to(edge_messages.device)
    dst_states = target_node_states.index_select(1, dst_index)
    dst_states = _align_last_dim(dst_states, edge_messages.shape[-1])

    edge_logits = (edge_messages * dst_states).sum(dim=-1) / safe_temperature
    local_flow = []
    log_local_flow = []
    for batch_idx in range(edge_logits.shape[0]):
        batch_flow = softmax(
            edge_logits[batch_idx], dst_index, num_nodes=target_node_states.shape[1]
        )
        local_flow.append(batch_flow)
        log_local_flow.append(torch.log(batch_flow.clamp(min=eps)))

    return {
        "edge_logits": edge_logits,
        "local_flow": torch.stack(local_flow, dim=0),
        "log_local_flow": torch.stack(log_local_flow, dim=0),
    }


def _prepare_source_log_weights(
    source_weights: torch.Tensor, num_nodes: int, eps: float = 1e-12
) -> torch.Tensor:
    source_weights = source_weights.reshape(-1).float()
    if source_weights.numel() != num_nodes:
        raise ValueError(
            f"source_weights size {source_weights.numel()} does not match "
            f"num_nodes={num_nodes}"
        )

    positive_mask = source_weights > 0
    if not bool(positive_mask.any()):
        raise ValueError("source_weights must contain at least one positive entry")

    normalized = source_weights.clamp(min=0)
    normalized = normalized / normalized[positive_mask].sum().clamp(min=eps)

    source_log_weights = torch.full(
        (num_nodes,), float("-inf"), device=source_weights.device
    )
    source_log_weights[positive_mask] = torch.log(
        normalized[positive_mask].clamp(min=eps)
    )
    return source_log_weights


def _scatter_logsumexp(
    values: torch.Tensor, index: torch.Tensor, dim_size: int
) -> torch.Tensor:
    output = torch.full((dim_size,), float("-inf"), device=values.device)
    unique_index = torch.unique(index)
    for node_idx in unique_index.tolist():
        mask = index == node_idx
        output[node_idx] = torch.logsumexp(values[mask], dim=0)
    return output


def forward_dp_from_local_flows(
    layer_traces: list[dict[str, torch.Tensor]],
    num_nodes: int,
    source_weights: torch.Tensor,
    eps: float = 1e-12,
) -> dict[str, list[torch.Tensor] | torch.Tensor]:
    source_log_weights = _prepare_source_log_weights(source_weights, num_nodes, eps)
    log_masses = [source_log_weights]

    current_log_mass = source_log_weights
    for trace in layer_traces:
        edge_index = trace["edge_index"]
        src_index = edge_index[0].to(current_log_mass.device)
        dst_index = edge_index[1].to(current_log_mass.device)
        log_local_flow = trace["log_local_flow"].to(current_log_mass.device)

        candidate_scores = current_log_mass.index_select(0, src_index) + log_local_flow
        next_log_mass = _scatter_logsumexp(candidate_scores, dst_index, num_nodes)
        log_masses.append(next_log_mass)
        current_log_mass = next_log_mass

    masses = [torch.exp(layer_log_mass) for layer_log_mass in log_masses]
    return {
        "source_log_weights": source_log_weights,
        "log_masses": log_masses,
        "masses": masses,
    }


def viterbi_decode_paths(
    layer_traces: list[dict[str, torch.Tensor]],
    num_nodes: int,
    source_weights: torch.Tensor,
    target_nodes: torch.Tensor,
    eps: float = 1e-12,
    topk_paths: int = 1,
) -> dict[int, dict[str, object]]:
    source_log_weights = _prepare_source_log_weights(source_weights, num_nodes, eps)
    beam_size = max(int(topk_paths), 1)
    beam_scores = torch.full(
        (num_nodes, beam_size), float("-inf"), device=source_log_weights.device
    )
    beam_scores[:, 0] = source_log_weights
    backpointers: list[dict[str, torch.Tensor]] = []

    for trace in layer_traces:
        edge_index = trace["edge_index"]
        edge_type = trace["edge_type"]
        src_index = edge_index[0].to(beam_scores.device)
        dst_index = edge_index[1].to(beam_scores.device)
        log_local_flow = trace["log_local_flow"].to(beam_scores.device)

        candidate_scores = beam_scores.index_select(0, src_index) + log_local_flow.unsqueeze(
            -1
        )
        next_beam_scores = torch.full(
            (num_nodes, beam_size), float("-inf"), device=beam_scores.device
        )
        prev_nodes = torch.full(
            (num_nodes, beam_size), -1, dtype=torch.long, device=beam_scores.device
        )
        prev_relations = torch.full(
            (num_nodes, beam_size), -1, dtype=torch.long, device=beam_scores.device
        )
        prev_ranks = torch.full(
            (num_nodes, beam_size), -1, dtype=torch.long, device=beam_scores.device
        )

        unique_targets = torch.unique(dst_index)
        for target_node in unique_targets.tolist():
            mask = dst_index == target_node
            node_scores = candidate_scores[mask]
            if node_scores.numel() == 0:
                continue
            flat_scores = node_scores.reshape(-1)
            finite_mask = torch.isfinite(flat_scores)
            if not bool(finite_mask.any()):
                continue
            finite_scores = flat_scores[finite_mask]
            finite_indices = finite_mask.nonzero(as_tuple=True)[0]
            keep = min(beam_size, int(finite_scores.shape[0]))
            top_scores, top_pos = torch.topk(finite_scores, k=keep)
            chosen_indices = finite_indices[top_pos]
            local_edge_indices = torch.div(
                chosen_indices, beam_size, rounding_mode="floor"
            )
            chosen_prev_ranks = chosen_indices % beam_size
            masked_src = src_index[mask]
            masked_rel = edge_type[mask]
            next_beam_scores[target_node, :keep] = top_scores
            prev_nodes[target_node, :keep] = masked_src[local_edge_indices]
            prev_relations[target_node, :keep] = masked_rel[local_edge_indices]
            prev_ranks[target_node, :keep] = chosen_prev_ranks

        backpointers.append(
            {
                "prev_nodes": prev_nodes,
                "prev_relations": prev_relations,
                "prev_ranks": prev_ranks,
            }
        )
        beam_scores = next_beam_scores

    decoded_paths: dict[int, dict[str, Any]] = {}
    for target_node in target_nodes.reshape(-1).tolist():
        target_scores = beam_scores[int(target_node)]
        paths: list[dict[str, Any]] = []
        for rank_idx, score_tensor in enumerate(target_scores.tolist()):
            score_log = float(score_tensor)
            if not math.isfinite(score_log):
                continue
            path: list[tuple[int, int, int]] = []
            current_node = int(target_node)
            current_rank = int(rank_idx)
            for layer_idx in range(len(backpointers), 0, -1):
                pointer = backpointers[layer_idx - 1]
                prev_node = int(pointer["prev_nodes"][current_node, current_rank].item())
                relation = int(pointer["prev_relations"][current_node, current_rank].item())
                prev_rank = int(pointer["prev_ranks"][current_node, current_rank].item())
                if prev_node < 0 or relation < 0 or prev_rank < 0:
                    path = []
                    break
                path.append((prev_node, current_node, relation))
                current_node = prev_node
                current_rank = prev_rank
            path.reverse()
            if not path:
                continue
            paths.append(
                {
                    "path": path,
                    "score_log": score_log,
                    "score": math.exp(score_log) if math.isfinite(score_log) else 0.0,
                    "rank": len(paths),
                }
            )

        best_path = paths[0] if paths else {"path": [], "score_log": float("-inf"), "score": 0.0}
        decoded_paths[int(target_node)] = {
            "path": best_path["path"],
            "score_log": float(best_path["score_log"]),
            "score": float(best_path["score"]),
            "best_hop": len(layer_traces),
            "paths": paths,
        }

    return decoded_paths
