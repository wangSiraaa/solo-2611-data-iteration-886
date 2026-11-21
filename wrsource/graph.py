"""Same-origin graph: transitive closure over named undirected evidence.

A *same-origin relation* in one evidence revision is the transitive closure of
the still-valid undirected evidence edges.  This module is deliberately pure
(no I/O); ``wrsource.evidence`` owns persistence and revision history.

Rules from the task:

* Evidence is *named*.  The same unordered pair of row identities may carry
  several independent pieces of evidence; revoking one does not disconnect the
  pair while another stays valid.
* A row whose body has been retracted stays a graph node and may still connect
  two other origins (the X--M--B example where M's body is withdrawn).
* Two records with identical bodies are not automatically same-origin: only
  explicit evidence creates edges.
* Snapshots are immutable views of one revision, so reopening an old revision
  after later revocations gives the old closure back exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Sequence


class OriginPath(NamedTuple):
    """A BFS witness path between two row identities."""

    nodes: List[str]
    evidence_ids: List[str]

    def as_dict(self) -> dict:
        return {"nodes": list(self.nodes),
                "evidence_ids": list(self.evidence_ids)}


@dataclass(frozen=True)
class ActiveEdge:
    """One still-valid piece of undirected evidence (endpoints normalised)."""

    evidence_id: str
    a: str
    b: str

    def other(self, node: str) -> str:
        if node == self.a:
            return self.b
        if node == self.b:
            return self.a
        raise KeyError(node)


class GraphSnapshot:
    """Immutable adjacency/closure view of the edges valid at one revision."""

    def __init__(self, revision: int, edges: Sequence[ActiveEdge]):
        self.revision = revision
        self._adj: Dict[str, List[ActiveEdge]] = {}
        for edge in edges:
            self._adj.setdefault(edge.a, []).append(edge)
            self._adj.setdefault(edge.b, []).append(edge)
        # Deterministic BFS: visit each node's neighbours in (other endpoint,
        # evidence id) order, so the witness path for a given revision never
        # varies between calls or processes.
        for node, at in self._adj.items():
            at.sort(key=lambda e: (e.other(node), e.evidence_id))
        self._components: Optional[Dict[str, int]] = None

    # -- basic views -----------------------------------------------------
    def nodes(self) -> List[str]:
        """All row identities participating in at least one valid edge."""
        return sorted(self._adj)

    def has_node(self, node: str) -> bool:
        return node in self._adj

    def edges_at(self, node: str) -> List[ActiveEdge]:
        return list(self._adj.get(node, ()))

    def all_edges(self) -> List[ActiveEdge]:
        unique = {id(e): e for at in self._adj.values() for e in at}
        return sorted(unique.values(),
                      key=lambda e: (e.a, e.b, e.evidence_id))

    def edge_count(self) -> int:
        return len({id(e) for at in self._adj.values() for e in at})

    # -- connected components (the transitive closure) -------------------
    def _component_map(self) -> Dict[str, int]:
        if self._components is None:
            comp: Dict[str, int] = {}
            cid = 0
            for start in sorted(self._adj):
                if start in comp:
                    continue
                comp[start] = cid
                stack = [start]
                while stack:
                    cur = stack.pop()
                    for edge in self._adj[cur]:
                        nxt = edge.other(cur)
                        if nxt not in comp:
                            comp[nxt] = cid
                            stack.append(nxt)
                cid += 1
            self._components = comp
        return self._components

    def component_id(self, node: str) -> Optional[int]:
        """Component number, or None for a node with no valid edge."""
        return self._component_map().get(node)

    def same_origin(self, x: str, y: str) -> bool:
        if x == y:
            return True
        cx, cy = self.component_id(x), self.component_id(y)
        return cx is not None and cx == cy

    def component_of(self, node: str) -> List[str]:
        cid = self.component_id(node)
        if cid is None:
            return []
        return sorted(n for n, c in self._component_map().items() if c == cid)

    def connected_exposures(self, node: str,
                            exposed: Sequence[str]) -> List[str]:
        """Exposed row identities sharing a component with ``node``.

        The node itself is included when it is directly exposed.  Isolated
        nodes only match when they themselves are in the exposed set.
        """
        exposed_set = set(exposed)
        cid = self.component_id(node)
        if cid is None:
            return [node] if node in exposed_set else []
        cmap = self._component_map()
        return sorted(e for e in exposed_set if cmap.get(e) == cid)

    # -- witness paths ----------------------------------------------------
    def find_path(self, source: str, target: str) -> Optional[OriginPath]:
        """BFS witness including every evidence id used, or None."""
        if source == target:
            return OriginPath([source], [])
        if source not in self._adj or target not in self._adj:
            return None
        prev_node: Dict[str, str] = {source: source}
        prev_edge: Dict[str, ActiveEdge] = {}
        queue = [source]
        head = 0
        while head < len(queue):
            cur = queue[head]
            head += 1
            for edge in self._adj[cur]:
                nxt = edge.other(cur)
                if nxt in prev_node:
                    continue
                prev_node[nxt] = cur
                prev_edge[nxt] = edge
                if nxt == target:
                    queue = []
                    break
                queue.append(nxt)
        if target not in prev_node:
            return None
        nodes: List[str] = [target]
        evidence: List[str] = []
        while nodes[-1] != source:
            cur = nodes[-1]
            evidence.append(prev_edge[cur].evidence_id)
            nodes.append(prev_node[cur])
        nodes.reverse()
        evidence.reverse()
        return OriginPath(nodes, evidence)
