# smgt
Enforcing Symbolic Constraints in Graph Transformers via Low-Rank Logic Adapters

This repository demonstrates how to enforce symbolic constraints in Graph Transformers using a unified adapter interface that supports both Logic Adapters and LoRA (Low-Rank Adaptation) adapters. The core implementation shows how to combine these approaches to maintain both model performance and constraint satisfaction, with the added benefit of lazy loading capabilities.

## Core Implementation

### logic_and_lora_demo.py

This script implements a novel approach to enforcing symbolic constraints in Graph Transformers through a unified adapter interface. The implementation focuses on three key aspects:

1. **Symbolic Constraint Enforcement**: 
   The system enforces type-based constraints by constructing a symbolic mask that prevents attention between forbidden node type pairs. For example, if we want to prevent Drug nodes from attending to Gene nodes, we create a mask where these positions are set to negative infinity. This mask is added to the attention scores before the softmax operation, effectively forcing the attention weights to zero for forbidden pairs. The mathematical formulation uses a mask $M_{ij}$ that is $-\infty$ for forbidden pairs and $0$ otherwise, ensuring that $\alpha'_{ij} = \mathrm{softmax}_j(e_{ij} + M_{ij})$ is exactly zero for prohibited interactions.

2. **Low-Rank Decomposition**:
   To make the approach scalable, we decompose the dense constraint mask into low-rank factors using Singular Value Decomposition (SVD). This decomposition, represented as $M \approx U_r \Sigma_r V_r^\top = A B^\top$, significantly reduces memory requirements from $O(N^2)$ to $O(Nr)$, where $r$ is a small rank parameter. The factors $A$ and $B$ are computed from the SVD components, with $A = U_r \sqrt{\Sigma_r}$ and $B = V_r \sqrt{\Sigma_r}$, providing an efficient approximation of the full constraint mask.

3. **Unified Adapter Interface**:
   Both Logic and LoRA adapters follow a common interface, making it easy to combine different types of constraints and adaptations. The Logic Adapter modifies attention scores based on symbolic rules, while the LoRA Adapter fine-tunes the model's weight matrices. This unified approach allows for flexible deployment and combination of different constraint types.

#### Implementation Example

The demo implements a simple but illustrative example using a heterogeneous graph with three node types:

1. **Graph Structure**:
   - 6 nodes with types: [Protein, Drug, Protein, Gene, Drug, Protein]
   - Each node has an 8-dimensional feature vector
   - The graph represents a simplified biological interaction network

2. **Constraint Rule**:
   - Single rule: Drug nodes cannot attend to Gene nodes
   - This represents a biological constraint where certain interactions are impossible
   - The rule is enforced through the Logic Adapter

3. **Logic Adapter in Action**:
   ```python
   # Create and initialize the adapter
   logic = LogicAdapter(rank=2)
   logic.load_rules(types, [(1, 2)])  # Drug(1) -> Gene(2) forbidden
   
   # Attach to model
   model.attach_adapter(logic)
   ```
   - The adapter creates a low-rank approximation of the constraint mask
   - During inference, it modifies attention scores to enforce the rule
   - The rank parameter (default=2) controls the approximation quality

4. **Training Process**:
   - First, train a baseline model without constraints
   - Then attach the Logic Adapter to enforce rules
   - Optionally add LoRA adapters for task-specific fine-tuning
   - The final model combines both constraint enforcement and task adaptation

5. **Results**:
   - Baseline model: May assign attention to forbidden Drug→Gene pairs
   - With Logic Adapter: Strictly enforces zero attention for forbidden pairs
   - With LoRA: Maintains constraints while adapting to the task
   - Combined: Achieves both constraint satisfaction and task performance

#### Key Components

- **Unified Adapter Interface**:
   ```python
   class BaseAdapter(nn.Module):
       def forward(self, x: torch.Tensor) -> torch.Tensor:
           raise NotImplementedError
   ```
   - Common interface for both Logic and LoRA adapters
   - Enables hot-swapping adapters at inference time
   - Supports combining multiple adapters

- **Logic Adapter Implementation**:
   ```python
   class LogicAdapter(BaseAdapter):
       def __init__(self, rank: int = 2):
           self.rank = rank
           self.A = None
           self.B = None
           self.initialized = False
       
       def load_rules(self, types: torch.Tensor, rules: list[tuple[int, int]]):
           # Initialize adapter parameters from rules
   ```
   - Lazy loading of rules during inference
   - Configurable rank for memory efficiency
   - Type-safe implementation

- **Heterogeneous Graph Structure**:
   - 6-node graph with 3 entity types: Protein (0), Drug (1), Gene (2)
   - Enforces a critical rule: Drug→Gene connections are forbidden
   - Measures constraint satisfaction through a violation rate metric that counts the proportion of forbidden pairs with non-zero attention

#### Training Pipeline

1. Train baseline Graph-Transformer classifier
2. Create and load Logic Adapter with rules:
   ```python
   logic = LogicAdapter(rank=2)
   logic.load_rules(types, RULES)
   model.attach_adapter(logic)
   ```
3. Fine-tune with LoRA adapters while maintaining constraints
4. Combine both adapters for optimal performance

#### Technical Details

- **Constraint Enforcement**: Uses a violation metric to measure constraint satisfaction
- **Low-Rank Decomposition**: Implements efficient matrix decomposition for symbolic masks
- **Parameter Efficiency**: Achieves constraint satisfaction with minimal additional parameters
- **Modular Design**: Provides reusable components through unified adapter interface
- **Type Safety**: Implements type hints for better code reliability

## Visualization Tool

### attention_vis_black.py

A supplementary visualization tool that generates attention pattern heatmaps to compare baseline and logic-masked attention. Outputs two PNG files:
- `baseline_black.png` - Baseline attention patterns
- `logic_black.png` - Attention patterns with logic masking

## Requirements

- Python 3.x
- PyTorch
- NumPy
- Matplotlib

## Usage

To run the core implementation:
```bash
python logic_and_lora_demo.py
```

To generate attention visualizations:
```bash
python attention_vis_black.py
```

## Output

The `logic_and_lora_demo.py` script will print:
- Baseline violation metrics
- Logic adapter violation metrics (after loading rules)
- LoRA adapter performance metrics
- Combined adapter results

## License

MIT License
