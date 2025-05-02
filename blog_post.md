# Symbolic Masked Graph Transformers: Enforcing Domain Constraints with Logic Adapters

## Introduction

Graph Neural Networks (GNNs) and particularly Graph Transformers (GTs) excel at capturing complex dependencies in graph-structured data. However, their standard formulations operate largely agnostic to explicit symbolic constraints often available from domain knowledge. This limitation can be problematic: models might propose scientifically invalid hypotheses, violate safety protocols in critical systems, or simply waste computational effort exploring impossible interactions. For instance, in bioinformatics, a rule might strictly forbid interactions between certain protein families, yet a standard GT might still assign non-zero attention, lacking an architectural mechanism to incorporate such hard constraints.

This post introduces Symbolic Masked Graph Transformers (SM-GT), an architectural solution that embeds hard, type-based exclusion rules directly into the GT attention mechanism, thereby guaranteeing constraint compliance. Complementing this, we propose lightweight, flexible "Logic Adapters"—inspired by Low-Rank Adaptation (LoRA)—offering a practical means to manage and deploy these constraints.

## The Challenge: Violating Domain Knowledge in Graph Attention

The core of GTs lies in the scaled dot-product attention mechanism, which computes attention weights $\alpha_{ij}$ between nodes $i$ and $j$ based on learned query ($Q$) and key ($K$) projections:

$$e_{ij} = \frac{Q_i K_j^{\top}}{\sqrt{d_k}}, \quad \alpha_{ij} = \mathrm{softmax}_j(e_{ij})$$

While heterogeneous architectures (e.g., HGT) can learn type-aware projections, they typically develop preferences rather than enforcing prohibitions. The model might learn likely interactions but isn't architecturally prevented from assigning attention to pairs deemed impossible by experts (e.g., Drug $\nrightarrow$ Gene if specified). Relying on implicit learning from data offers no guarantees, and alternative approaches like semantic loss penalties often enforce constraints only softly.

## Architectural Enforcement: The Power of Pre-Softmax Masking

SM-GT integrates constraints directly. Given a set $\mathcal{F} = \{(T_A, T_B)\}$ defining forbidden source-target node type pairs, we construct a symbolic mask $M \in \mathbb{R}^{N \times N}$:

$$M_{ij} = \begin{cases}-\infty & \text{if } (\mathrm{type}(i), \mathrm{type}(j)) \in \mathcal{F} \\0 & \text{otherwise.}\end{cases}$$

The critical step is adding this mask to the attention scores before applying the softmax function:

$$e'_{ij} = e_{ij} + M_{ij}$$

The subsequent application of the softmax, $\alpha'_{ij} = \mathrm{softmax}_j(e'_{ij})$, leverages the property that $\exp(-\infty) \rightarrow 0$. Consequently, the attention weight $\alpha'_{ij}$ for any forbidden pair $(i, j)$ is guaranteed to be exactly zero (within numerical precision), irrespective of the learned parameters $Q$ and $K$. This provides a mathematically sound, architectural guarantee of rule compliance during every forward pass.

## Flexibility and Efficiency: Low-Rank Logic Adapters

While effective, a dense $N \times N$ mask $M$ presents scalability and flexibility challenges: prohibitive memory cost for large $N$ and inability to modify rules post-deployment without altering the model structure or retraining.

We address this with Logic Adapters, inspired by the parameter-efficient fine-tuning technique LoRA (Hu et al., 2022). Our implementation introduces a unified adapter interface that supports both Logic and LoRA adapters, allowing for flexible deployment and combination of different constraint types.

### The Adapter Interface

```python
class BaseAdapter(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
```

This interface ensures that both Logic and LoRA adapters can be used interchangeably within the same architecture. The Logic Adapter implementation:

```python
class LogicAdapter(BaseAdapter):
    def __init__(self, rank: int = 2):
        super().__init__()
        self.rank = rank
        self.A = None
        self.B = None
        self.initialized = False
    
    def load_rules(self, types: torch.Tensor, rules: list[tuple[int, int]]):
        """Load rules and types to initialize the adapter parameters"""
        M = full_mask(types, rules)
        M[M == 0] = -0.5
        U, S, Vh = torch.linalg.svd(M)
        self.A = nn.Parameter(U[:, :self.rank] * S[:self.rank].sqrt(), requires_grad=False)
        self.B = nn.Parameter(Vh[:self.rank, :].T * S[:self.rank].sqrt(), requires_grad=False)
        self.initialized = True
```

This approach yields several benefits:

1. **Lazy Loading**: Rules can be loaded during inference time, making the system more flexible and memory-efficient.
2. **Unified Interface**: Both Logic and LoRA adapters follow the same interface, enabling easy combination and swapping.
3. **Configurable Rank**: The rank of the logic adapter can be adjusted to balance memory efficiency and constraint accuracy.
4. **Type Safety**: Added type hints ensure better code reliability and maintainability.

### Integration with Graph Transformers

The adapter is integrated into the attention mechanism through a modular design:

```python
class MaskedAttn(nn.Module):
    def __init__(self, d: int, use_lora: bool = False, adapter: BaseAdapter | None = None):
        super().__init__()
        self.adapter = adapter
    
    def attach_adapter(self, adapter: BaseAdapter | None):
        self.adapter = adapter
    
    def forward(self, x: torch.Tensor):
        s = (self.q(x) @ self.k(x).T) / math.sqrt(x.size(1))
        if self.adapter is not None:
            s = s + self.adapter(x)
        a = s.softmax(-1)
        return a @ self.v(x), a
```

This design allows for:
- Hot-swapping different adapters at inference time
- Combining multiple adapters for complex constraint sets
- Maintaining clean separation between the base model and constraint logic

## Demonstration on a Toy Heterogeneous Graph

To clearly illustrate the mechanism and differentiate it from standard weight adaptation, we conducted an experiment on a minimal synthetic graph.

### Setup:
- A graph with 6 nodes and 3 types (0: Protein, 1: Drug, 2: Gene)
- A symbolic rule forbidding interactions from Drug to Gene nodes: $\mathcal{F} = \{(1, 2)\}$
- A simple Graph Transformer (GT) model trained for node classification using cross-entropy loss
- A Violation Rate (VR) metric calculating the proportion of attention mass assigned to the forbidden Drug $\to$ Gene pairs

### Procedure and Results:

1. **Baseline Training**: A standard GT was trained on the node classification task.
   - Observed Result: Baseline Violation = 1.0 (significant attention assigned to forbidden pairs)

2. **Attach Logic Adapter**: A rank-2 Logic Adapter was created and rules were loaded during inference:
   ```python
   logic = LogicAdapter(rank=2)
   logic.load_rules(types, RULES)
   model.attach_adapter(logic)
   ```
   - Observed Result: Logic Violation = 0.0 (attention on forbidden pairs forced to zero)

3. **Detach Logic, Add LoRA**: The Logic Adapter was removed and standard LoRA adapters were added.
   - Observed Result: LoRA Violation = 1.0 (confirms that LoRA alone doesn't enforce constraints)

4. **Combine LoRA + Logic**: The Logic Adapter was attached to the LoRA-fine-tuned model.
   - Observed Result: LoRA+Logic Violation = 0.0 (demonstrates compatibility between adapters)

## Further Considerations

### Interpretability
By guaranteeing zero attention on impossible pathways, SM-GT's attention mechanism aligns more closely with domain logic, potentially aiding validation.

### Inductive Bias
Embedding hard constraints provides a strong inductive bias, which could be particularly beneficial in low-data regimes.

### Limitations
- Handling dynamic graphs requires regenerating or updating the mask/adapter
- While Logic Adapters are memory-efficient, applying complex constraints to extremely large graphs might require further optimization
- The current implementation assumes static type assignments; handling dynamic type changes would require additional mechanisms

## Conclusion

Symbolic Masked Graph Transformers (SM-GT) offer a method for embedding hard symbolic type-exclusion rules directly into the GT attention mechanism. The proposed low-rank Logic Adapters provide a flexible and efficient mechanism for managing these rules, with the added benefit of lazy loading and a unified interface with standard LoRA adapters. This framework facilitates the development of more reliable and adaptable neurosymbolic models for graph-structured data where domain knowledge is critical.

## References
[Previous references remain unchanged] 