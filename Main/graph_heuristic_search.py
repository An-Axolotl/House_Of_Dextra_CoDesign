#!/usr/bin/env python3
"""
Graph Heuristic Search implementation following Algorithm 2 from RoboGrammar paper.
Core algorithm implementation with Graph Neural Network for design value prediction.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import pickle
import json
import time
from hand_groups import HandGroup
import copy

import math
def _gumbel_noise():
    u = random.random()
    return -math.log(-math.log(max(u, 1e-12)))

# Fingertip vocabulary (fixed order for one-hot encoding)
FINGERTIP_CHOICES = ["standard", "thinner", "wedged"]
TIP2IDX = {name: i for i, name in enumerate(FINGERTIP_CHOICES)}

class GraphNeuralNetwork(nn.Module):
    """Simple GNN for design value estimation"""
    
    def __init__(self, node_features: int = 70, hidden_dim: int = 128, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        
        # Node embedding layers
        self.node_embed = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.ReLU(),
            self.dropout,  # Add dropout
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Message passing layers  
        self.message_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                self.dropout,  # Add dropout
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_layers)
        ])
        
        # Global pooling and output
        self.global_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, node_features: torch.Tensor, edge_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_features: [num_nodes, node_features]
            edge_indices: [2, num_edges] - source and target node indices
        Returns:
            Scalar value prediction for the graph
        """
        # Embed nodes
        h = self.node_embed(node_features)  # [num_nodes, hidden_dim]
        
        # Message passing
        for layer in self.message_layers:
            messages = []
            for i in range(h.size(0)):
                # Get neighbors of node i
                neighbors = edge_indices[1][edge_indices[0] == i]
                if len(neighbors) > 0:
                    neighbor_features = h[neighbors]  # [num_neighbors, hidden_dim]
                    node_features_expanded = h[i].unsqueeze(0).expand(len(neighbors), -1)
                    combined = torch.cat([node_features_expanded, neighbor_features], dim=1)
                    message = layer(combined).mean(dim=0)  # Aggregate messages
                else:
                    message = torch.zeros_like(h[i])
                messages.append(message)
            h = torch.stack(messages)
        
        # Global pooling
        graph_embedding = h.mean(dim=0)  # Simple mean pooling
        value = self.global_pool(graph_embedding)
        return value.squeeze()
    
@dataclass
class DesignGraph:
    """Represents a hand design as a graph structure"""
    nodes: List[Dict]  # Node features (finger type, servo presence, etc.)
    edges: List[Tuple[int, int]]  # Edge connections
    terminals: Set[int]  # Terminal node indices (fully specified)
    non_terminals: Set[int]  # Non-terminal node indices (need expansion)
    design_string: str  # String representation for hashing/comparison
    group: Optional[str] = None
    
    def __hash__(self):
        return hash(self.design_string)
    
    def __eq__(self, other):
        return isinstance(other, DesignGraph) and self.design_string == other.design_string
    
    def __len__(self):
        return len(self.nodes)
    
    def is_complete(self) -> bool:
        """Check if design is fully specified (no non-terminals)"""
        return len(self.non_terminals) == 0
    
    def one_hot(self, i, K):
            v = [0.0]*K
            if 0 <= i < K: v[i] = 1.0
            return v
    
    def to_tensor_data(self, group_name: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert to tensors for GNN with group conditioning"""
        # Group one-hot encoding (6 dimensions for 6 groups)
        GROUPS = [g.value for g in HandGroup]  # ["sym3", "sym4", "sym5", "anth21", "anth27", "anth33"]
        group_oh = [0.0] * len(GROUPS)
        if group_name in GROUPS:
            group_oh[GROUPS.index(group_name)] = 1.0
        
        node_tensor = []
        for node in self.nodes:
            fid = int(node.get('finger_id', 0))
            is_base = 1.0 if node.get('is_base', fid == 0) else 0.0
            is_term = 1.0 if node.get('is_terminal', False) else 0.0
            s = int(node.get('servo_count', 0))
            g1 = float(node.get('grammar_1_count', 0))
            g2 = float(node.get('grammar_2_count', 0))
            
            g1n = (g1 - 1.0) / 9.0 if g1 > 0 else 0.0
            g2n = (g2 - 1.0) / 9.0 if g2 > 0 else 0.0
            
            present = 1.0 if (s > 0 or fid == 0) else 0.0
            s_oh = self.one_hot(s, 4)
            pos_oh = self.one_hot(max(0, fid), 6)
            actionable = 1.0 if (present and not node.get('is_terminal', False) and fid > 0) else 0.0
            
            tip = node.get('fingertip_type', 'standard')
            tip_idx = TIP2IDX.get(str(tip), 0)
            tip_oh = self.one_hot(tip_idx, len(TIP2IDX))
            if present == 0.0:
                tip_oh = [0.0] * len(TIP2IDX)
            
            # Combine features: base (19) + group context (6) = 25 active dims
            feats = [
                is_base, is_term, present, actionable, g1n, g2n,  # 6
                *s_oh,      # 4
                *pos_oh,    # 6
                *tip_oh,    # 3
                *group_oh   # 6
            ]
            
            # Pad to 70 dimensions (increased from 64)
            while len(feats) < 70:
                feats.append(0.0)
            
            node_tensor.append(feats)
        
        node_features = torch.tensor(node_tensor, dtype=torch.float32)
        
        if self.edges:
            edge_indices = torch.tensor(self.edges, dtype=torch.long).t()
            rev = edge_indices.flip(0)
            edge_indices = torch.cat([edge_indices, rev], dim=1)
        else:
            edge_indices = torch.zeros((2, 0), dtype=torch.long)
        
        return node_features, edge_indices
    
def canonicalize_fingers(design: "DesignGraph") -> "DesignGraph":
    """
    Produce a safe canonical 6-node layout while PRESERVING finalization:
    - node 0 is palm (terminal, base)
    - exactly 5 finger slots (1..5)
    - active fingers (servo_count >= 2) first, then ghosts (servo_count < 2)
    - edges rebuilt as star: (0,i) for i=1..5
    - *explicit is_terminal stays true* even if servo_count >= 2
    """
    
    nodes = list(design.nodes) if getattr(design, "nodes", None) else []

    # Ensure a palm at index 0
    if not nodes:
        nodes = [{'finger_id': 0, 'is_base': True, 'is_terminal': True}]
    else:
        if nodes[0].get('finger_id', 0) != 0:
            n0 = copy.deepcopy(nodes[0])
            n0['finger_id'] = 0
            n0['is_base'] = True
            n0['is_terminal'] = True
            nodes[0] = n0
        else:
            nodes[0] = copy.deepcopy(nodes[0])
            nodes[0]['finger_id'] = 0
            nodes[0]['is_base'] = True
            nodes[0]['is_terminal'] = True

    # Trim/pad to palm + 5 fingers
    if len(nodes) > 6:
        nodes = nodes[:6]

    while len(nodes) < 6:
        idx = len(nodes)
        nodes.append({
            'finger_id': idx,
            'grammar_1_count': 0,
            'grammar_2_count': 0,
            'servo_count': 0,
            'fingertip_type': 'standard',
            'is_terminal': True,   # ghosts are terminal
            'is_base': False,
        })

    # Order: active (>=2 servos) then ghosts
    active_idx = [i for i in range(1, 6) if nodes[i].get('servo_count', 0) >= 2]
    ghost_idx  = [i for i in range(1, 6) if nodes[i].get('servo_count', 0) <  2]
    ordered = [0] + active_idx + ghost_idx

    # Map old->new indices
    idx_map = {old_i: new_fid for new_fid, old_i in enumerate(ordered)}

    # Rebuild nodes in canonical order, PRESERVING explicit is_terminal
    new_nodes = []
    for new_fid, old_i in enumerate(ordered):
        old = copy.deepcopy(nodes[old_i])
        n = {}
        n.update(old)
        n['finger_id'] = new_fid
        if new_fid == 0:
            n['is_base'] = True
            n['is_terminal'] = True
        else:
            servo = int(old.get('servo_count', 0))
            was_terminal = bool(old.get('is_terminal', False))
            n['is_base'] = False
            # Preserve explicit terminal OR ghost-ness
            n['is_terminal'] = was_terminal or (servo < 2)
        new_nodes.append(n)

    # Star edges from palm
    new_edges = [(0, i) for i in range(1, 6)]

    # Remap terminal/non-terminal sets from original, then enforce ghost terminals
    orig_terms = set(getattr(design, 'terminals', set()))
    orig_non   = set(getattr(design, 'non_terminals', set()))

    new_terminals = { idx_map.get(i, i) for i in orig_terms if idx_map.get(i, i) < 6 }
    new_non_terms = { idx_map.get(i, i) for i in orig_non   if idx_map.get(i, i) < 6 }

    # Ensure ghosts are terminal; finalized fingers are NOT in non_terminals
    for i in range(1, 6):
        servo = new_nodes[i].get('servo_count', 0)
        if servo < 2:
            new_terminals.add(i)
            if i in new_non_terms:
                new_non_terms.remove(i)
        if new_nodes[i].get('is_terminal', False):
            if i in new_non_terms:
                new_non_terms.remove(i)
            new_terminals.add(i)

    # Palm is always terminal
    new_terminals.add(0)
    if 0 in new_non_terms:
        new_non_terms.remove(0)

    canon = DesignGraph(
        nodes=new_nodes,
        edges=new_edges,
        terminals=new_terminals,
        non_terminals=new_non_terms,
        design_string=f"{getattr(design,'design_string','design')}_canon",
        group=design.group
    )

    # Preserve evaluated/effective annotations
    for attr in ("_effective_key", "_effective_group", "_effective_graph"):
        if hasattr(design, attr):
            setattr(canon, attr, getattr(design, attr))
    if hasattr(design, "evaluated_group"):
        canon.evaluated_group = getattr(design, "evaluated_group")

    return canon

class SequentialFingerSchedule:
    """Finger 1→2→3→4→5 OR 5→4→3→2→1; per finger phases: servo -> (g1 if servo==3) -> g2 -> tip -> finalize."""
    def __init__(self, design: DesignGraph, phases: List[str], *, direction: str = "forward"):
        self.phases = phases
        base_order = [f for f in [1,2,3,4,5] if f in design.non_terminals]
        if direction == "backward":
            base_order = list(reversed(base_order))
        self.order = base_order
        self.phase_idx = {f: 0 for f in self.order}
        self.f_idx = 0  # which finger in self.order is current


    def current(self) -> Optional[Tuple[int, str]]:
        # Move to next unfinished finger
        while self.f_idx < len(self.order) and self.phase_idx[self.order[self.f_idx]] >= len(self.phases):
            self.f_idx += 1
        if self.f_idx >= len(self.order):
            return None
        f = self.order[self.f_idx]
        return f, self.phases[self.phase_idx[f]]

    def advance_for(self, finger: int, auto_steps: int = 1):
        if finger in self.phase_idx:
            self.phase_idx[finger] = min(self.phase_idx[finger] + auto_steps, len(self.phases))

    def done(self) -> bool:
        return self.f_idx >= len(self.order)

class HandDesignGenerator:
    """Generates hand designs following grammar rules"""
    
    def __init__(self, max_fingers: int = 5, max_servos_per_finger: int = 4):
        self.max_fingers = max_fingers
        self.max_servos_per_finger = max_servos_per_finger
        
        # Grammar rules
        self.grammar_rules = {
            'set_grammar_1': self._set_grammar_1_rule,
            'set_grammar_2': self._set_grammar_2_rule, 
            'set_servo_count': self._set_servo_count_rule,
            'set_fingertip_type': self._set_fingertip_type_rule,
            'finalize_finger': self._finalize_finger_rule
        }

    @staticmethod
    def _choose_finger_idx(design: DesignGraph, params: Optional[Dict], *, min_servo: int = 0) -> Optional[int]:
        candidates = [
            i for i in design.non_terminals
            if i > 0 and int(design.nodes[i].get("servo_count", 0)) >= min_servo
        ]
        if not candidates:
            return None
        if params and "finger_idx" in params:
            return int(params["finger_idx"])
        return random.choice(candidates)

    @staticmethod
    def _clone_with_finger_updates(
        design: DesignGraph,
        finger_idx: int,
        *,
        suffix: str,
        updates: Dict,
        terminals: Optional[Set[int]] = None,
        non_terminals: Optional[Set[int]] = None,
    ) -> DesignGraph:
        new_nodes = design.nodes.copy()
        new_nodes[finger_idx] = new_nodes[finger_idx].copy()
        new_nodes[finger_idx].update(updates)
        return DesignGraph(
            new_nodes,
            design.edges,
            design.terminals if terminals is None else terminals,
            design.non_terminals if non_terminals is None else non_terminals,
            design.design_string + suffix,
            group=design.group,
        )
        
    def get_param_options_for_rule(self, design: DesignGraph, rule: str) -> List[Dict]:
        is_anth = (design.group in ("anth21", "anth27", "anth33"))
        active_slots = [i for i in design.non_terminals if i > 0]  # fingers that can still change
        opts: List[Dict] = []

        if rule == 'set_servo_count':
            for f in active_slots:
                if is_anth and f == 1:
                    opts.append({'finger_idx': f, 'servo_count': 3})  # thumb lock
                else:
                    for s in (2, 3):
                        opts.append({'finger_idx': f, 'servo_count': s})

        elif rule == 'set_grammar_1':
            for f in active_slots:
                for c in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10):
                    opts.append({'finger_idx': f, 'grammar_1_count': c})

        elif rule == 'set_grammar_2':
            for f in active_slots:
                for c in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10):
                    opts.append({'finger_idx': f, 'grammar_2_count': c})

        elif rule == 'set_fingertip_type':
            for f in active_slots:
                for tip in FINGERTIP_CHOICES:
                    opts.append({'finger_idx': f, 'fingertip_type': tip})

        elif rule == 'finalize_finger':
            # Optional: only allow finalize on active fingers (>=2 servos) to avoid wasting moves
            for f in active_slots:
                if design.nodes[f].get('servo_count', 0) >= 2:
                    opts.append({'finger_idx': f})
        return opts

    PHASES = ["set_servo_count", "set_grammar_1", "set_grammar_2", "set_fingertip_type", "finalize_finger"]

    def options_for_phase(self, design: DesignGraph, finger_idx: int, phase: str) -> List[Dict]:
        """Small legal option set for this finger+phase, with dependencies enforced."""
        if finger_idx == 0 or finger_idx not in design.non_terminals:
            return []

        is_anth = (design.group in ("anth21", "anth27", "anth33"))
        servo = int(design.nodes[finger_idx].get("servo_count", 0))
        opts: List[Dict] = []

        if phase == "set_servo_count":
            if is_anth and finger_idx == 1:
                opts = [{'finger_idx': finger_idx, 'servo_count': 3}]
            else:
                opts = [{'finger_idx': finger_idx, 'servo_count': s} for s in (2, 3)]

        elif phase == "set_grammar_1":
            # Only meaningful when servo==3; otherwise force 0 and auto-skip.
            if servo == 3:
                opts = [{'finger_idx': finger_idx, 'grammar_1_count': c} for c in range(1, 11)]
            else:
                # Return a single forced option (0) so caller can auto-apply+skip this phase.
                opts = [{'finger_idx': finger_idx, 'grammar_1_count': 0}]

        elif phase == "set_grammar_2":
            # Always meaningful for active fingers (servo in {2,3}); choose 1..10
            if servo >= 2:
                opts = [{'finger_idx': finger_idx, 'grammar_2_count': c} for c in range(1, 11)]
            else:
                opts = []

        elif phase == "set_fingertip_type":
            opts = [{'finger_idx': finger_idx, 'fingertip_type': tip} for tip in FINGERTIP_CHOICES]

        elif phase == "finalize_finger":
            if servo >= 2:
                opts = [{'finger_idx': finger_idx}]
            else:
                opts = []

        return opts

    
    def get_initial_design(self, group: Optional[HandGroup] = None) -> DesignGraph:
        """Create initial design respecting group constraints"""

        # Determine active finger count based on group
        if group is None:
            # No group specified - random 3-5
            active_count = random.randint(3, 5)
        elif group == HandGroup.SYM3:
            active_count = 3
        elif group == HandGroup.SYM4:
            active_count = 4
        elif group == HandGroup.SYM5:
            active_count = 5
        elif group in [HandGroup.ANTH21, HandGroup.ANTH27, HandGroup.ANTH33]:
            active_count = 5
        else:
            active_count = random.randint(3, 5)
        
        # Palm node
        nodes = [{'finger_id': 0, 'is_base': True, 'is_terminal': True}]
        
        # Add active fingers 1..active_count
        for i in range(1, active_count + 1):
            servo = random.choice([2, 3])
            if group in [HandGroup.ANTH21, HandGroup.ANTH27, HandGroup.ANTH33] and i == 1:
                servo = 3  # thumb locked to 3
            nodes.append({
                'finger_id': i,
                'grammar_1_count': 0,
                'grammar_2_count': random.randint(1, 10),
                'servo_count': servo,
                'is_terminal': False,
                'is_base': False,
                'fingertip_type': random.choice(FINGERTIP_CHOICES),
            })
        # Ghosts (only if any left)
        for i in range(active_count + 1, 6):
            nodes.append({
                'finger_id': i,
                'grammar_1_count': 0,
                'grammar_2_count': 0,
                'servo_count': 0,
                'is_terminal': True,
                'is_base': False,
                'fingertip_type': 'standard',
            })
        
        edges = [(0, i) for i in range(1, 6)]  # Connect all fingers to base
        terminals = {0} | {i for i in range(active_count + 1, 6)}  # Palm + ghost fingers are terminal
        non_terminals = {i for i in range(1, active_count + 1)}    # Only active fingers need expansion
        
        gstr = group.value if group else "nogroup"
        dg = DesignGraph(nodes, edges, terminals, non_terminals, f"base_{active_count}active_{gstr}_canonical", group=gstr)
        return dg
    
    def get_available_rules(self, design: DesignGraph) -> List[str]:
        """Include servo_count in available rules"""
        if not any(i > 0 for i in design.non_terminals):
            return []
        return self.PHASES.copy()
    
    def apply_rule(self, design: DesignGraph, rule: str, params: Optional[Dict] = None) -> DesignGraph:
        if rule not in self.grammar_rules:
            raise ValueError(f"Unknown rule: {rule}")
        out = self.grammar_rules[rule](design, params)  # pass params through
        out.group = design.group
        # canon = canonicalize_fingers(out)
        # canon.group = design.group
        return out
    
    def _set_grammar_1_rule(self, design: DesignGraph, params: Optional[Dict] = None) -> DesignGraph:
        finger_idx = self._choose_finger_idx(design, params)
        if finger_idx is None:
            return design
        new_count = params.get('grammar_1_count') if params and 'grammar_1_count' in params else random.randint(1, 10)
        return self._clone_with_finger_updates(
            design,
            finger_idx,
            suffix=f"_g1f{finger_idx}c{new_count}",
            updates={"grammar_1_count": int(new_count)},
        )
    
    def _set_grammar_2_rule(self, design: DesignGraph, params: Optional[Dict] = None) -> DesignGraph:
        finger_idx = self._choose_finger_idx(design, params)
        if finger_idx is None:
            return design
        new_count = params.get('grammar_2_count') if params and 'grammar_2_count' in params else random.randint(1, 10)
        return self._clone_with_finger_updates(
            design,
            finger_idx,
            suffix=f"_g2f{finger_idx}c{new_count}",
            updates={"grammar_2_count": int(new_count)},
        )

    def _set_servo_count_rule(self, design: DesignGraph, params: Optional[Dict] = None) -> DesignGraph:
        finger_idx = self._choose_finger_idx(design, params)
        if finger_idx is None:
            return design

        is_anth = (design.group in ("anth21", "anth27", "anth33"))

        if is_anth and finger_idx == 1:
            new_count = 3  # thumb lock
        else:
            proposed = params.get('servo_count') if params and 'servo_count' in params else random.choice([2, 3])
            # Hard guard: never accept 0 here even if a caller passes it.
            if proposed not in (2, 3):
                proposed = 2
            new_count = proposed

        return self._clone_with_finger_updates(
            design,
            finger_idx,
            suffix=f"_s{finger_idx}c{new_count}",
            updates={"servo_count": int(new_count)},
        )
        
        
    def _set_fingertip_type_rule(self, design: DesignGraph, params: Optional[Dict] = None) -> DesignGraph:
        finger_idx = self._choose_finger_idx(design, params)
        if finger_idx is None:
            return design
        tip = params.get('fingertip_type') if params and 'fingertip_type' in params else random.choice(FINGERTIP_CHOICES)
        return self._clone_with_finger_updates(
            design,
            finger_idx,
            suffix=f"_tipf{finger_idx}{tip}",
            updates={"fingertip_type": tip},
        )

    
    def _finalize_finger_rule(self, design: DesignGraph, params: Optional[Dict] = None) -> DesignGraph:
        finger_idx = self._choose_finger_idx(design, params, min_servo=2)
        if finger_idx is None:
            return design

        new_term = design.terminals.copy()
        new_term.add(finger_idx)
        new_non = design.non_terminals.copy()
        new_non.remove(finger_idx)
        return self._clone_with_finger_updates(
            design,
            finger_idx,
            suffix=f"_fin{finger_idx}",
            updates={"is_terminal": True},
            terminals=new_term,
            non_terminals=new_non,
        )

class GraphHeuristicSearch:
    """Main algorithm implementation following RoboGrammar Algorithm 2"""
    
    def __init__(self, 
                 num_iterations: int = 100,
                 num_candidates: int = 10,
                 opt_iterations: int = 50,
                 batch_size: int = 8,
                 epsilon: float = 0.1,
                 learning_rate: float = 1e-3,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 debug: bool = False):
        
        self.debug = debug
        
        self.N = num_iterations
        self.M = num_candidates
        self.K = num_candidates  # Number of designs to sample per iteration
        self.opt_iter = opt_iterations
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.device = device
        self.num_iterations = num_iterations
        self.num_candidates = num_candidates
        
        # Initialize components
        self.generator = HandDesignGenerator()
        self.value_network = GraphNeuralNetwork().to(device)
        self.optimizer = optim.Adam(self.value_network.parameters(), lr=learning_rate)
        
        # Initialize lookup table and best design tracking
        self.lookup_table: Dict[Tuple[str, str], float] = {}
        self.best_design: Optional[DesignGraph] = None
        self.best_reward: float = 0.0
        
        # Store all seen designs for training
        self.seen_designs: List[DesignGraph] = []
        
        # Placeholder evaluator - will be replaced by specific implementation
        self.evaluator = None
    
    def evaluate_design_value(self, design: DesignGraph) -> float:
        """Evaluate design using value network with group context."""
        
        assert self.evaluator is not None, "Set self.evaluator before evaluation"
        
        d = canonicalize_fingers(design)
        grp_obj = self.evaluator._current_group()
        grp = grp_obj.value if grp_obj else "default"

        # 1) Prefer an existing effective key
        if hasattr(d, "_effective_key") and d._effective_key:
            key = d._effective_key
        else:
            # 2) Compute the *provisional* effective key for the current group
            key = self.compute_effective_key_for_group(d, grp_obj)

        # Lookup ONLY by effective key
        cached = self.lookup_table.get((key, grp))
        if cached is not None and cached > -999.0:
            return cached

        # Evaluate with GNN
        node_features, edge_indices = d.to_tensor_data(group_name=grp)
        node_features = node_features.to(self.device)
        edge_indices = edge_indices.to(self.device)

        with torch.no_grad():
            self.value_network.eval()
            value = self.value_network(node_features, edge_indices).item()

        self.value_network.train()
        return value

    def _score_partial(self, design: DesignGraph, group_name: str, cache: Dict[str, float]) -> float:
        key = f"p:{group_name}:{design.design_string}"
        if key in cache:
            return cache[key]
        node_features, edge_indices = design.to_tensor_data(group_name=group_name)
        node_features = node_features.to(self.device)
        edge_indices = edge_indices.to(self.device)
        with torch.no_grad():
            self.value_network.eval()
            v = self.value_network(node_features, edge_indices).item()
        self.value_network.train()
        cache[key] = v
        return v
    
    def generate_candidate_designs(self) -> List[DesignGraph]:
        assert self.evaluator is not None, "Set self.evaluator before generation"
        unique: List[DesignGraph] = []
        seen_eff: set[str] = set()
        batch_path_tabu: set = set()

        # Plan diversity across the K slots
        plans = []
        for k in range(self.K):
            direction = "forward" if (k % 2 == 0) else "backward"
            rng_seed = random.randrange(1_000_000_000)
            noise_scale = 0.38 + 0.02 * k   # small ramp
            plans.append((direction, rng_seed, noise_scale))

        for direction, seed, noise in plans:
            cand = self._generate_greedy_design(
                beam=1, direction=direction, rng_seed=seed,
                noise_scale=noise, batch_path_tabu=batch_path_tabu
            )

            grp_obj = self.evaluator._current_group()
            eff_key = self.compute_effective_key_for_group(cand, grp_obj)

            if eff_key in seen_eff:
                # One local re-roll with boosted noise; no while-loop roulette
                cand = self._generate_greedy_design(
                    beam=1, direction=direction, rng_seed=random.randrange(1_000_000_000),
                    noise_scale=noise * 1.8, batch_path_tabu=batch_path_tabu
                )
                eff_key = self.compute_effective_key_for_group(cand, grp_obj)

            if eff_key not in seen_eff:
                seen_eff.add(eff_key)
                unique.append(cand)
            elif self.debug:
                print(f"[dedupe] collided eff_key={eff_key}")

        if self.debug and len(unique) < self.K:
            print(f"[warn] produced {len(unique)} unique (target {self.K}). Consider slightly higher noise_scale (e.g., +0.05).")

        return unique



    def _generate_greedy_design(
        self,
        beam: int = 1,
        *,
        direction: str = "forward",
        rng_seed: int = 0,
        noise_scale: float = 0.45,
        batch_path_tabu: Optional[set] = None,
    ) -> DesignGraph:
        if rng_seed:
            random.seed(rng_seed)

        grp_obj = self.evaluator._current_group()
        grp = grp_obj.value if grp_obj else "default"

        cur = self.generator.get_initial_design(group=grp_obj)
        setattr(cur, "applied", set())
        cache: Dict[str, float] = {}

        sched = SequentialFingerSchedule(cur, self.generator.PHASES, direction=direction)
        MAX_STEPS = 60
        step = 0
        path_key = []  # list of per-step decision tuples

        while step < MAX_STEPS and not sched.done() and not cur.is_complete():
            item = sched.current()
            if item is None:
                break
            finger, phase = item

            if (finger, phase) in getattr(cur, "applied", set()):
                sched.advance_for(finger, auto_steps=1)
                continue

            options = self.generator.options_for_phase(cur, finger, phase)
            if not options:
                sched.advance_for(finger, auto_steps=1)
                continue

            # Forced option (e.g., g1==0 when servo==2)
            if len(options) == 1 and (phase == "set_grammar_1" and options[0].get("grammar_1_count", None) == 0):
                cur = self.generator.apply_rule(cur, phase, {**options[0], 'finger_idx': finger})
                cur.applied.add((finger, phase))
                sched.advance_for(finger, auto_steps=1)
                step += 1
                continue

            # Build successors; collect compact decision keys
            succs, succ_keys = [], []
            for opt in options:
                s2 = self.generator.apply_rule(cur, phase, {**opt, 'finger_idx': finger})
                s2.applied = set(getattr(cur, "applied", set()))
                s2.applied.add((finger, phase))
                key_tuple = (finger, phase) + tuple(sorted((k, str(v)) for k, v in opt.items()))
                succs.append(s2)
                succ_keys.append(key_tuple)

            # Score with Gumbel noise (diversifies deterministically greedy rollout)
            base_scores = [self._score_partial(s2, grp, cache) for s2 in succs]
            noisy_scores = [s + noise_scale * _gumbel_noise() for s in base_scores]

            # Pick best successor whose path-prefix isn't tabu for this batch
            picked = None
            for idx in sorted(range(len(succs)), key=lambda i: -noisy_scores[i]):
                tentative_path = tuple(path_key + [succ_keys[idx]])
                if batch_path_tabu is not None and tentative_path in batch_path_tabu:
                    continue
                picked = succs[idx]
                path_key.append(succ_keys[idx])
                if batch_path_tabu is not None:
                    batch_path_tabu.add(tuple(path_key))
                break

            if picked is None:
                # Rare fallback: accept top even if tabu
                best_i = max(range(len(succs)), key=lambda i: noisy_scores[i])
                picked = succs[best_i]
                path_key.append(succ_keys[best_i])
                if batch_path_tabu is not None:
                    batch_path_tabu.add(tuple(path_key))

            cur = picked

            # advance logic
            if phase == "set_servo_count":
                if int(cur.nodes[finger].get("servo_count", 0)) == 2:
                    if cur.nodes[finger].get("grammar_1_count", 0) != 0:
                        cur = self.generator.apply_rule(cur, "set_grammar_1", {'finger_idx': finger, 'grammar_1_count': 0})
                    cur.applied.add((finger, "set_grammar_1"))
                    sched.advance_for(finger, auto_steps=2)
                else:
                    sched.advance_for(finger, auto_steps=1)
            else:
                sched.advance_for(finger, auto_steps=1)

            step += 1

        return cur
    
    def get_partial_ancestors(self, design: DesignGraph) -> List[DesignGraph]:
        """Get all partial ancestor designs"""
        ancestors = []
        
        # For now, just return some partial versions of the design
        for i in range(1, len(design.nodes)):
            partial_nodes = design.nodes[:i+1]
            partial_edges = [(s, t) for s, t in design.edges if s <= i and t <= i]
            partial_terminals = {idx for idx in design.terminals if idx <= i}
            partial_non_terminals = {idx for idx in design.non_terminals if idx <= i}
            
            partial_design = DesignGraph(
                partial_nodes, partial_edges, partial_terminals,
                partial_non_terminals, f"{design.design_string}_partial_{i}", group=design.group
            )
            ancestors.append(partial_design)
        
        return ancestors
    
    def update_lookup_table(self, design: DesignGraph, reward: float, group: Optional[str]):
        grp = group or "default"
        d = canonicalize_fingers(design)

        grp_obj = None
        if self.evaluator:
            try:
                grp_obj = [g for g in HandGroup if g.value == grp][0]
            except IndexError:
                pass

        eff_key = getattr(d, "_effective_key", None)
        if not eff_key and grp_obj:
            eff_key = self.compute_effective_key_for_group(d, grp_obj)

        if eff_key:
            self.lookup_table[(eff_key, grp)] = max(self.lookup_table.get((eff_key, grp), float('-inf')), float(reward))
            # Ensure the complete graph itself is available to train on
            d.evaluated_group = grp
            self.seen_designs.append(d)

        if grp_obj:
            for anc in self.get_partial_ancestors(d):
                anc_eff = getattr(anc, "_effective_key", None) or self.compute_effective_key_for_group(anc, grp_obj)
                self.lookup_table[(anc_eff, grp)] = max(self.lookup_table.get((anc_eff, grp), float('-inf')), float(reward))
                # NEW: also train on partials (throttle to avoid blow-up)
                if random.random() < 0.5:  # keep ~half; tune as needed
                    anc.evaluated_group = grp
                    self.seen_designs.append(anc)



    def train_value_network(self):
        """Train the value network using seen designs with per-group training."""
        if len(self.seen_designs) < self.batch_size:
            return

        # Group designs by their evaluated_group
        designs_by_group = {}
        for d in self.seen_designs:
            grp = getattr(d, 'evaluated_group', None) or "default"
            designs_by_group.setdefault(grp, []).append(d)
        
        # Filter out groups with insufficient data
        trainable_groups = {
            grp: designs for grp, designs in designs_by_group.items()
            if len(designs) >= self.batch_size
        }
        
        if not trainable_groups:
            print("⚠️  No groups have enough data for training")
            return
        
        print(f"🔧 Training GNN on {len(trainable_groups)} groups:")
        for grp, designs in trainable_groups.items():
            print(f"   - {grp}: {len(designs)} designs")
        
        # Train on each group separately
        for group, group_designs in trainable_groups.items():
            # Get valid training data for this group
            train_set = []
            train_values = []
            for d in group_designs:
                eff_key = getattr(d, "_effective_key", None)
                if not eff_key and self.evaluator:
                    grp_obj = [g for g in HandGroup if g.value == group][0]
                    eff_key = self.compute_effective_key_for_group(d, grp_obj)
                if eff_key is None:
                    continue
                v = self.lookup_table.get((eff_key, group))

                if v is None or v <= -999.0:
                    continue
                train_set.append(d)
                train_values.append(v)
            
            if len(train_set) < self.batch_size:
                continue
            
            print(f"\n   [{group}] Training on {len(train_set)} designs")
            
            # Training loop for this group
            group_opt_iter = max(self.opt_iter // len(trainable_groups), 20)  # Distribute iterations
            
            for step in range(group_opt_iter):
                batch_designs = random.sample(train_set, min(self.batch_size, len(train_set)))
                
                batch_loss = 0.0
                for design in batch_designs:
                    d = canonicalize_fingers(design)
                    eff_key = getattr(d, "_effective_key", None)
                    if eff_key is None and self.evaluator:
                        grp_obj = [g for g in HandGroup if g.value == group][0]
                        eff_key = self.compute_effective_key_for_group(d, grp_obj)
                    if eff_key is None:
                        continue
                    target_value = self.lookup_table.get((eff_key, group), 0.0)
                    
                    # Convert to tensors with group context
                    node_features, edge_indices = d.to_tensor_data(group_name=group)
                    node_features = node_features.to(self.device)
                    edge_indices = edge_indices.to(self.device)
                    
                    # Add noise for regularization (less in later steps)
                    if step < group_opt_iter // 2:
                        noise = torch.randn_like(node_features) * 0.01
                        node_features = node_features + noise
                    
                    predicted_value = self.value_network(node_features, edge_indices)
                    target_tensor = torch.tensor(target_value, dtype=torch.float32, device=self.device)
                    
                    loss = (predicted_value - target_tensor) ** 2
                    batch_loss += loss
                
                # L2 regularization
                l2_lambda = 0.001
                l2_reg = sum(p.pow(2.0).sum() for p in self.value_network.parameters())
                
                batch_loss = batch_loss / len(batch_designs) + l2_lambda * l2_reg
                
                self.optimizer.zero_grad()
                batch_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                if step % 10 == 0:
                    print(f"      Step {step}/{group_opt_iter}: loss={batch_loss.item():.4f}")
        
        # Learning rate decay (once per full training call)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.99
            print(f"   Learning rate decayed to: {param_group['lr']:.6f}")

    def compute_effective_key_for_group(self, design, group_obj):
        assert self.evaluator is not None, "Set self.evaluator before evaluation"
        d = canonicalize_fingers(design)
        finger_codes, mode, thumb_slot, finger_tips = self.evaluator.effective_codes_for(d, group_obj)
        thumb_str = f"_{thumb_slot}" if thumb_slot is not None else ""
        tip_pairs = ",".join(f"{c}:{t}" for c, t in zip(finger_codes, finger_tips))
        return f"eff_{mode}{thumb_str}_{tip_pairs}"
    
    def run(self) -> Tuple[Optional[DesignGraph], float]:
        """Run the Graph Heuristic Search with detailed progress tracking"""
        print(f"\n🚀 STARTING GRAPH HEURISTIC SEARCH")
        print(f"   Iterations: {self.num_iterations}")
        print(f"   Candidates per iteration: {self.num_candidates}")
        print(f"   Total designs to evaluate: {self.num_iterations * self.num_candidates}")
        print(f"   Device: {self.device}")
        
        search_start_time = time.time()
        
        # Store initial epsilon for annealing
        initial_epsilon = self.epsilon
        final_epsilon = 0.05  # End with less exploration
        
        for iteration in range(self.num_iterations):
            # Anneal epsilon over iterations
            progress = iteration / max(1, self.num_iterations - 1)
            self.epsilon = initial_epsilon + (final_epsilon - initial_epsilon) * progress
            
            iter_start_time = time.time()
            
            print(f"\n🔄 ITERATION {iteration + 1}/{self.num_iterations}")
            print(f"   Epsilon: {self.epsilon:.3f}")  # Show current epsilon
            print(f"   Designs seen so far: {len(self.seen_designs)}")
            print(f"   Current best score: {self.best_reward:.3f}")
            
            # Generate candidates
            print(f"   📋 Generating {self.num_candidates} candidate designs...")
            candidates = self.generate_candidate_designs()
            print(f"   ✓ Generated {len(candidates)} candidates")
            
            # Evaluate candidates in parallel
            print(f"   🔄 Canonicalizing candidates...")
            canonical_candidates = [canonicalize_fingers(candidate) for candidate in candidates]

            current_group = (
                self.evaluator._current_group().value
                if hasattr(self, "evaluator") and hasattr(self.evaluator, "_current_group")
                else "default"
            )

            print(f"   🎯 Evaluating {len(canonical_candidates)} candidates in parallel...")
            parallel_start_time = time.time()

            iteration_scores = self.evaluator.evaluate_designs_parallel(canonical_candidates)

            parallel_time = time.time() - parallel_start_time
            print(f"   ✓ Parallel evaluation completed in {parallel_time:.1f}s")

            # --- Process results ---
            for i, (candidate, score) in enumerate(zip(canonical_candidates, iteration_scores)):
                print(f"   📊 Candidate {i+1}: {candidate.design_string} -> Score: {score:.3f}")

                # Set the group before adding to seen_designs
                candidate.evaluated_group = current_group
                self.seen_designs.append(candidate)

                # Lookup table is keyed by (design_string, group)
                self.update_lookup_table(candidate, score, candidate.evaluated_group)

                # Best-tracking
                if score > self.best_reward:
                    self.best_reward = score
                    self.best_design = candidate
                    print(f"      🏆 NEW BEST DESIGN! Score: {score:.3f}")
            
            iter_time = time.time() - iter_start_time
            avg_score = sum(iteration_scores) / len(iteration_scores) if iteration_scores else 0
            
            print(f"\n   📊 Iteration {iteration + 1} Summary:")
            print(f"      Time: {iter_time:.1f}s")
            print(f"      Average score: {avg_score:.3f}")
            print(f"      Best score this iter: {max(iteration_scores) if iteration_scores else -1000:.3f}")
            print(f"      Overall best: {self.best_reward:.3f}")
            
            self.train_value_network()
            
            # Save checkpoint every 5 iterations
            if (iteration + 1) % 5 == 0:
                self.save_checkpoint(iteration + 1)
                print(f"      💾 Checkpoint saved for iteration {iteration+1}")

        
        total_time = time.time() - search_start_time
        
        print(f"\n🎉 GRAPH HEURISTIC SEARCH COMPLETED!")
        print(f"   Total time: {total_time/60:.1f} minutes")
        print(f"   Designs evaluated: {len(self.seen_designs)}")
        print(f"   Best score: {self.best_reward:.3f}")
        if self.best_design:
            print(f"   Best design: {self.best_design.design_string}")
        
        return self.best_design, self.best_reward

    def save_checkpoint(self, iteration: int, save_dir: str = "ghs_checkpoints"):
        """Save current state"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        checkpoint = {
            'iteration': iteration,
            'value_network_state': self.value_network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'lookup_table': self.lookup_table,
            'best_design': self.best_design,
            'best_reward': self.best_reward,
            'seen_designs': self.seen_designs
        }
        
        torch.save(checkpoint, os.path.join(save_dir, f"checkpoint_{iteration}.pt"))
        
        # Save best design separately
        if self.best_design:
            with open(os.path.join(save_dir, f"best_design_{iteration}.json"), 'w') as f:
                json.dump({
                    'design_string': self.best_design.design_string,
                    'reward': self.best_reward,
                    'nodes': self.best_design.nodes,
                    'edges': self.best_design.edges
                }, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load saved state"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.value_network.load_state_dict(checkpoint['value_network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.lookup_table = checkpoint['lookup_table']
        self.best_design = checkpoint['best_design']
        self.best_reward = checkpoint['best_reward']
        self.seen_designs = checkpoint['seen_designs']
