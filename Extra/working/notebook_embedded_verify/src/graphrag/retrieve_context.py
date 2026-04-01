"""Inspired by RAG (Lewis et al., 2020), StructGPT (Jiang et al., 2023), RoG (Luo et al., 2023),
and educational GraphRAG work comparing graph retrieval with standard RAG.

Original idea: retrieve structured context from a knowledge graph to ground LLM reasoning.
Simplified version here: retrieve a bounded candidate-centered subgraph plus KB profile, because education-focused
GraphRAG studies show that overly large graph neighborhoods can become noisy for fine-grained tasks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

from src.data.load_data import build_input_text
from src.ontology.graph_utils import build_candidate_subgraph, summarize_subgraph
from src.utils.io import read_json
from src.utils.text import lexical_overlap_score


class ContextRetriever:
    def __init__(self, kb_entries: list[dict[str, Any]], graph: nx.Graph):
        self.kb_entries = kb_entries
        self.graph = graph
        self.kb_by_id = {entry["misconception_id"]: entry for entry in kb_entries}

    @classmethod
    def from_artifacts(cls, kb_path: str | Path, graph_path: str | Path | None = None) -> "ContextRetriever":
        kb_entries = read_json(kb_path, default=[])
        if graph_path and Path(graph_path).exists():
            graph = nx.read_gexf(graph_path)
        else:
            graph = nx.Graph()
            for entry in kb_entries:
                graph.add_node(entry["misconception_id"], misconception_name=entry.get("misconception_name", ""))
        return cls(kb_entries=kb_entries, graph=graph)

    def get_candidate_entry(self, candidate_id: str) -> dict[str, Any]:
        if candidate_id in self.kb_by_id:
            return self.kb_by_id[candidate_id]
        return {
            "misconception_id": candidate_id,
            "misconception_name": candidate_id,
            "family": "",
            "short_description": "",
            "prerequisite_concepts": [],
            "trigger_patterns": [],
            "formal_counterexample": "",
            "remediation_hint": "",
            "aliases": [],
            "concept_tags": [],
            "cognitive_roots": [],
            "evidence_examples": [],
        }

    def retrieve(self, row: pd.Series | dict[str, Any], candidate_id: str, top_k_neighbors: int = 3, max_trigger_matches: int = 3) -> dict[str, Any]:
        if not isinstance(row, pd.Series):
            row = pd.Series(row)
        entry = self.get_candidate_entry(candidate_id)
        row_text = build_input_text(row)
        trigger_score, trigger_matches = lexical_overlap_score(row_text, entry.get("trigger_patterns", []))
        concept_terms = list(entry.get("concept_tags", [])) + list(entry.get("cognitive_roots", [])) + list(entry.get("prerequisite_concepts", []))
        concept_score, concept_matches = lexical_overlap_score(row_text, concept_terms)

        max_subgraph_nodes = max(2, int(top_k_neighbors) + 1)
        subgraph = build_candidate_subgraph(self.graph, candidate_id, max_nodes=max_subgraph_nodes)
        subgraph_meta = summarize_subgraph(subgraph, candidate_id)

        neighbors = []
        if candidate_id in subgraph:
            for neighbor_id in subgraph.neighbors(candidate_id):
                edge_data = subgraph.get_edge_data(candidate_id, neighbor_id) or {}
                neighbor_entry = self.get_candidate_entry(neighbor_id)
                neighbors.append(
                    {
                        "misconception_id": neighbor_id,
                        "misconception_name": neighbor_entry.get("misconception_name", neighbor_id),
                        "relations": edge_data.get("relations", []),
                        "shared_values": edge_data.get("shared_values", []),
                    }
                )
        neighbors = sorted(neighbors, key=lambda item: (len(item["relations"]), len(item["shared_values"])), reverse=True)[:top_k_neighbors]
        support_count = float(entry.get("support_count", 0) or 0)
        support_score = min(1.0, support_count / 4.0)
        ontology_support_score = min(1.0, 0.45 * trigger_score + 0.35 * concept_score + 0.20 * support_score)

        context = {
            "misconception_id": candidate_id,
            "misconception_name": entry.get("misconception_name", candidate_id),
            "family": entry.get("family", ""),
            "short_description": entry.get("short_description", ""),
            "prerequisite_concepts": entry.get("prerequisite_concepts", []) or [],
            "concept_tags": entry.get("concept_tags", []) or [],
            "cognitive_roots": entry.get("cognitive_roots", []) or [],
            "matched_trigger_patterns": trigger_matches[:max_trigger_matches],
            "matched_concept_terms": concept_matches[:max_trigger_matches],
            "formal_counterexample": entry.get("formal_counterexample", ""),
            "remediation_hint": entry.get("remediation_hint", ""),
            "neighbor_summaries": neighbors,
            "graph_neighbor_count": int(self.graph.degree(candidate_id)) if candidate_id in self.graph else 0,
            "trigger_match_count": len(trigger_matches[:max_trigger_matches]),
            "concept_match_count": len(concept_matches[:max_trigger_matches]),
            "trigger_score": trigger_score,
            "concept_score": concept_score,
            "ontology_support_score": ontology_support_score,
            "evidence_examples": list(entry.get("evidence_examples", []) or [])[:2],
            **subgraph_meta,
        }
        context["neighbor_relation_types"] = sorted({relation for neighbor in neighbors for relation in neighbor.get("relations", [])})
        context["context_block"] = self.format_context_block(context)
        return context

    def format_context_block(self, context: dict[str, Any]) -> str:
        neighbor_lines = []
        for neighbor in context.get("neighbor_summaries", []):
            relation_text = ", ".join(neighbor.get("relations", [])) or "related"
            shared_text = ", ".join(neighbor.get("shared_values", [])) or "no shared values"
            neighbor_lines.append(f"- {neighbor['misconception_name']} ({relation_text}; {shared_text})")
        evidence_lines = [f"- {example}" for example in context.get("evidence_examples", [])] or ["- none"]
        edge_lines = [f"- {item}" for item in context.get("edge_summaries", [])[:3]] or ["- none"]
        prerequisites = ", ".join(context.get("prerequisite_concepts", [])) or "none"
        tags = ", ".join(context.get("concept_tags", [])) or "none"
        roots = ", ".join(context.get("cognitive_roots", [])) or "none"
        triggers = " | ".join(context.get("matched_trigger_patterns", [])) or "none"
        concept_matches = " | ".join(context.get("matched_concept_terms", [])) or "none"
        neighbors = "\n".join(neighbor_lines) if neighbor_lines else "- none"
        evidence = "\n".join(evidence_lines)
        edges = "\n".join(edge_lines)
        return "\n".join(
            [
                f"Candidate family: {context.get('family', '') or 'unknown'}",
                f"Short description: {context.get('short_description', '') or 'none'}",
                f"Concept tags: {tags}",
                f"Cognitive roots: {roots}",
                f"Prerequisite concepts: {prerequisites}",
                f"Matched trigger patterns: {triggers}",
                f"Matched concept terms: {concept_matches}",
                f"Ontology support score: {context.get('ontology_support_score', 0.0):.3f}",
                f"Formal counterexample: {context.get('formal_counterexample', '') or 'none'}",
                f"Remediation hint: {context.get('remediation_hint', '') or 'none'}",
                "Observed example evidence:",
                evidence,
                f"Subgraph size: {context.get('subgraph_node_count', 0)} nodes, {context.get('subgraph_edge_count', 0)} edges, density={context.get('subgraph_density', 0.0):.3f}",
                "Subgraph edges:",
                edges,
                "Related misconceptions:",
                neighbors,
            ]
        )