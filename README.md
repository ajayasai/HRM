Refer README from https://github.com/sapientinc/HRM

Modifications made:

Replaced the regular attention with attention mechanism mentioned in this paper:

**RiemannFormer: A Framework for Attention in Curved Spaces** - https://arxiv.org/abs/2506.07405

Traditional Transformer Limitations

Transformers excel at capturing long-range dependencies through self-attention but lack inductive biases like locality and position-awareness, which are crucial in domains such as computer vision or on small datasets.

In Euclidean space, all token embeddings are considered to lie in a flat space. However, many types of data—especially sequences or images—exhibit more complex relationships that may be better modeled in curved (non-Euclidean) spaces.

📐 Core Concept: Curved Geometric Space
Manifold Hypothesis

Instead of representing tokens in a flat space, RiemannFormer assumes they lie on a Riemannian manifold—a curved geometric space with varying metric tensors that define distances and angles locally.

Each token embedding is viewed as residing at a point p on this manifold 𝑀 and the associated query and key vectors lie in the tangent space TpM

	​

	​


