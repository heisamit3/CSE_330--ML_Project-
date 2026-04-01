"""Inspired by StructGPT (Jiang et al., 2023), RoG (Luo et al., 2023), and educational GraphRAG studies.

Original idea: use graph-structured entities and relations to retrieve faithful context for reasoning.
Simplified version here: build a small transparent NetworkX graph from the misconception KB and extract
bounded candidate-centered subgraphs to avoid the noisy-context failure mode reported in educational GraphRAG work.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import networkx as nx

from src.utils.io import read_json, write_json
from src.utils.text import tokenize


MANUAL_RELATIONS = ("causes", "exacerbates", "shares_cognitive_root_with")


def _set_values(entry: dict[str, Any], field: str) -> set[str]:
    values = entry.get(field, []) or []
    if isinstance(values, str):
        values = [values]
    return {str(value).strip().lower() for value in values if str(value).strip()}


def _name_token_set(entry: dict[str, Any]) -> set[str]:
    parts = [entry.get("misconception_name", "")] + list(entry.get("aliases", []) or [])
    tokens = set()
    for part in parts:
        tokens |= set(tokenize(str(part)))
    return tokens


def _normalize_graph_links(entry: dict[str, Any]) -> dict[str, list[str]]:
    graph_links = entry.get("graph_links", {}) or {}
    normalized: dict[str, list[str]] = {relation: [] for relation in MANUAL_RELATIONS}
    for relation, targets in graph_links.items():
        if isinstance(targets, str):
            targets = [targets]
        normalized[relation] = [str(target).strip() for target in targets if str(target).strip()]
    return normalized


def _add_relation(graph: nx.Graph, left_id: str, right_id: str, relation: str, shared_values: list[str]) -> None:
    if left_id == right_id:
        return
    if graph.has_edge(left_id, right_id):
        payload = graph[left_id][right_id]
        payload.setdefault("relations", [])
        if relation not in payload["relations"]:
            payload["relations"].append(relation)
        payload.setdefault("shared_values", [])
        payload["shared_values"] = sorted(set(payload["shared_values"]) | set(shared_values))
    else:
        graph.add_edge(left_id, right_id, relations=[relation], shared_values=sorted(set(shared_values)))


def build_graph_from_kb(entries: list[dict[str, Any]]) -> nx.Graph:
    graph = nx.Graph()
    entry_by_id = {entry["misconception_id"]: entry for entry in entries if entry.get("misconception_id")}

    for entry in entries:
        node_id = entry["misconception_id"]
        graph.add_node(
            node_id,
            misconception_name=entry.get("misconception_name", node_id),
            family=entry.get("family", ""),
            short_description=entry.get("short_description", ""),
            trigger_patterns=" | ".join(entry.get("trigger_patterns", [])),
            prerequisite_concepts=" | ".join(entry.get("prerequisite_concepts", [])),
            concept_tags=" | ".join(entry.get("concept_tags", [])),
            cognitive_roots=" | ".join(entry.get("cognitive_roots", [])),
            support_count=entry.get("support_count", 0),
        )

    for entry in entries:
        source_id = entry["misconception_id"]
        for relation_name, target_ids in _normalize_graph_links(entry).items():
            for target_id in target_ids:
                if target_id in entry_by_id:
                    _add_relation(graph, source_id, target_id, relation_name, [relation_name])

    for left, right in combinations(entries, 2):
        left_id = left["misconception_id"]
        right_id = right["misconception_id"]
        shared_prereq = sorted(_set_values(left, "prerequisite_concepts") & _set_values(right, "prerequisite_concepts"))
        if shared_prereq:
            _add_relation(graph, left_id, right_id, "shares_prerequisite", shared_prereq)

        shared_trigger = sorted(_set_values(left, "trigger_patterns") & _set_values(right, "trigger_patterns"))
        if shared_trigger:
            _add_relation(graph, left_id, right_id, "related_trigger", shared_trigger)

        shared_roots = sorted(_set_values(left, "cognitive_roots") & _set_values(right, "cognitive_roots"))
        if shared_roots:
            _add_relation(graph, left_id, right_id, "shares_cognitive_root_with", shared_roots)

        shared_tags = sorted(_set_values(left, "concept_tags") & _set_values(right, "concept_tags"))
        if left.get("family", "") == right.get("family", "") and shared_tags:
            _add_relation(graph, left_id, right_id, "possible_confusion_with", shared_tags)

        left_tokens = _name_token_set(left)
        right_tokens = _name_token_set(right)
        overlap = sorted(left_tokens & right_tokens)
        if left.get("family", "") == right.get("family", "") and len(overlap) >= 2:
            _add_relation(graph, left_id, right_id, "alias_overlap", overlap)
    return graph


def build_candidate_subgraph(graph: nx.Graph, candidate_id: str, max_nodes: int = 6) -> nx.Graph:
    if candidate_id not in graph:
        return nx.Graph()
    selected = [candidate_id]
    scored_neighbors = []
    for neighbor_id in graph.neighbors(candidate_id):
        edge_data = graph.get_edge_data(candidate_id, neighbor_id) or {}
        score = (
            len(edge_data.get("relations", [])),
            len(edge_data.get("shared_values", [])),
            graph.degree(neighbor_id),
        )
        scored_neighbors.append((score, neighbor_id))
    scored_neighbors.sort(reverse=True)
    for _, neighbor_id in scored_neighbors[: max(0, max_nodes - 1)]:
        selected.append(neighbor_id)
    return graph.subgraph(selected).copy()


def summarize_subgraph(graph: nx.Graph, candidate_id: str) -> dict[str, Any]:
    relation_counter: Counter[str] = Counter()
    edge_summaries: list[str] = []
    for left_id, right_id, payload in graph.edges(data=True):
        relations = payload.get("relations", [])
        shared_values = payload.get("shared_values", [])
        for relation in relations:
            relation_counter[relation] += 1
        edge_summaries.append(
            f"{left_id} <-> {right_id} via {', '.join(relations) or 'related'} ({', '.join(shared_values[:3]) or 'no shared values'})"
        )
    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    density = float(nx.density(graph)) if node_count > 1 else 0.0
    return {
        "focus_node": candidate_id,
        "subgraph_node_count": node_count,
        "subgraph_edge_count": edge_count,
        "subgraph_density": density,
        "relation_counts": dict(relation_counter),
        "edge_summaries": edge_summaries,
    }


def graph_summary(graph: nx.Graph) -> dict[str, Any]:
    relation_counter: dict[str, int] = {}
    for _, _, payload in graph.edges(data=True):
        for relation in payload.get("relations", []):
            relation_counter[relation] = relation_counter.get(relation, 0) + 1
    degree_map = dict(graph.degree())
    top_degree_nodes = sorted(degree_map.items(), key=lambda item: item[1], reverse=True)[:5]
    return {
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "relations": relation_counter,
        "avg_degree": 0.0 if graph.number_of_nodes() == 0 else sum(degree_map.values()) / graph.number_of_nodes(),
        "top_degree_nodes": top_degree_nodes,
    }


def export_graph(graph: nx.Graph, graph_path: str | Path, summary_path: str | Path) -> dict[str, Any]:
    nx.write_gexf(graph, graph_path)
    summary = graph_summary(graph)
    write_json(summary, summary_path)
    return summary


def load_graph_from_kb(kb_path: str | Path, graph_path: str | Path | None = None) -> nx.Graph:
    entries = read_json(kb_path, default=[])
    graph = build_graph_from_kb(entries)
    if graph_path:
        nx.write_gexf(graph, graph_path)
    return graph