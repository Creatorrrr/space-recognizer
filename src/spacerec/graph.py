"""Scene graph: proximity + topological relations between world objects.

The world frame follows the first camera (OpenCV RDF), so -y is approximately
"up" for roughly level footage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import GraphCfg
from .objects import WorldObject


@dataclass
class GraphEdge:
    a: WorldObject
    b: WorldObject
    dist: float
    relation: str  # "above" (a가 b 위) | "beside"

    @property
    def label(self) -> str:
        symbol = {"above": "↑", "beside": "—"}[self.relation]
        return f"{symbol} {self.dist:.2f}"


def build_graph(objects: list[WorldObject], cfg: GraphCfg) -> list[GraphEdge]:
    edges: list[GraphEdge] = []
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            a, b = objects[i], objects[j]
            delta = b.position - a.position
            dist = float(np.linalg.norm(delta))
            if dist > cfg.near_dist:
                continue
            horiz = float(np.hypot(delta[0], delta[2]))
            dy = float(delta[1])  # +y는 아래 방향
            if abs(dy) > cfg.vertical_ratio * max(horiz, 1e-6):
                # a 위에 b가 있으면 dy<0 → above 관계는 항상 (위, 아래) 순서로 정규화
                top, bottom = (b, a) if dy < 0 else (a, b)
                edges.append(GraphEdge(top, bottom, dist, "above"))
            else:
                edges.append(GraphEdge(a, b, dist, "beside"))
    return edges
