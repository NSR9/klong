"""Create synthetic demo data for running the full pipeline."""
import json
from pathlib import Path

papers = [
    {
        "paper_id": "demo_paper_001",
        "title": "Attention-Based Feature Aggregation for Image Classification",
        "abstract": "We propose a novel attention mechanism for feature aggregation in CNNs.",
        "authors": ["Alice Researcher", "Bob Scientist"],
        "github_url": "https://github.com/example/attention-feature-agg",
        "pdf_url": "https://arxiv.org/pdf/2301.00001",
        "markdown": (
            "# Attention-Based Feature Aggregation for Image Classification\n\n"
            "## Abstract\n"
            "We propose a novel attention mechanism for feature aggregation in convolutional neural networks. "
            "Our method, Attention Feature Aggregation (AFA), applies channel and spatial attention to intermediate "
            "feature maps before global pooling. We demonstrate state-of-the-art results on CIFAR-10 (97.2%).\n\n"
            "## 1. Introduction\n"
            "Deep convolutional neural networks have achieved remarkable success in image classification. "
            "However, standard global average pooling discards spatial information.\n\n"
            "## 2. Method\n"
            "### 2.1 Channel Attention\n"
            "Given feature map F of shape (C, H, W), we compute channel attention weights:\n"
            "w_c = sigmoid(MLP(GAP(F)))\n\n"
            "### 2.2 Spatial Attention\n"
            "w_s = sigmoid(Conv(concat(max_pool(F), avg_pool(F))))\n\n"
            "### 2.3 Feature Aggregation\n"
            "Final features: z = GAP(w_c * w_s * F)\n\n"
            "## 3. Experiments\n"
            "- Dataset: CIFAR-10\n- Backbone: ResNet-18\n- Optimizer: SGD momentum 0.9\n"
            "- Learning rate: 0.1 cosine annealing\n- Epochs: 200\n- Batch size: 128\n\n"
            "| Method | CIFAR-10 |\n|--------|----------|\n"
            "| ResNet-18 | 95.3 |\n| + SE | 96.1 |\n| + AFA (ours) | 97.2 |\n\n"
            "## 4. Code\n```python\nimport torch\nimport torch.nn as nn\n\n"
            "class ChannelAttention(nn.Module):\n    def __init__(self, channels, reduction=16):\n"
            "        super().__init__()\n        self.mlp = nn.Sequential(\n"
            "            nn.Linear(channels, channels // reduction),\n            nn.ReLU(),\n"
            "            nn.Linear(channels // reduction, channels),\n            nn.Sigmoid()\n        )\n\n"
            "    def forward(self, x):\n        b, c, h, w = x.shape\n"
            "        y = x.mean(dim=(2, 3))\n        y = self.mlp(y).unsqueeze(-1).unsqueeze(-1)\n"
            "        return x * y\n```\n"
        ),
        "conference": "ICML",
        "year": 2024,
    },
    {
        "paper_id": "demo_paper_002",
        "title": "Efficient Gradient Compression for Distributed Training",
        "abstract": "We propose TopK gradient compression with error feedback.",
        "authors": ["Carol Distributed", "Dave Parallel"],
        "github_url": "https://github.com/example/grad-compress",
        "pdf_url": "https://arxiv.org/pdf/2301.00002",
        "markdown": (
            "# Efficient Gradient Compression for Distributed Training\n\n"
            "## Abstract\nCommunication overhead is a major bottleneck in distributed deep learning. "
            "We propose a TopK gradient compression scheme with error feedback.\n\n"
            "## 1. Method\n### 1.1 TopK Sparsification\n"
            "Keep only top-K elements of the gradient by magnitude.\n\n"
            "### 1.2 Error Feedback\nAccumulate compression error and add to next iteration.\n\n"
            "## 2. Experiments\n- Model: ResNet-50 on ImageNet\n- Workers: 8 GPUs\n"
            "- Compression ratio: 100x\n- Final accuracy: 76.1% (baseline 76.3%)\n\n"
            "## 3. Code\n```python\nimport torch\n\n"
            "def topk_compress(grad, ratio=0.01):\n"
            "    k = max(1, int(grad.numel() * ratio))\n"
            "    values, indices = torch.topk(grad.abs().flatten(), k)\n"
            "    compressed = torch.zeros_like(grad.flatten())\n"
            "    compressed[indices] = grad.flatten()[indices]\n"
            "    return compressed.view_as(grad)\n```\n"
        ),
        "conference": "NeurIPS",
        "year": 2024,
    },
    {
        "paper_id": "demo_paper_003",
        "title": "Learning Rate Warmup Revisited: A Simple Baseline",
        "abstract": "We revisit learning rate warmup and show linear warmup of 5 epochs is sufficient.",
        "authors": ["Eve Optimizer"],
        "github_url": "https://github.com/example/lr-warmup",
        "pdf_url": "https://arxiv.org/pdf/2301.00003",
        "markdown": (
            "# Learning Rate Warmup Revisited\n\n"
            "## Abstract\nLearning rate warmup is widely used but poorly understood. "
            "We find that a simple linear warmup of 5 epochs is sufficient.\n\n"
            "## 1. Experiments\n- Models: ResNet-18, ViT-S\n- Dataset: CIFAR-10\n"
            "- Warmup: linear, cosine, constant\n- Epochs: 100\n\n"
            "| Warmup | ResNet-18 | ViT-S |\n|--------|-----------|-------|\n"
            "| None | 94.1 | 92.3 |\n| Linear 5ep | 95.3 | 94.8 |\n\n"
            "## 2. Code\n```python\n"
            "def linear_warmup(optimizer, epoch, warmup_epochs, base_lr):\n"
            "    if epoch < warmup_epochs:\n"
            "        lr = base_lr * (epoch + 1) / warmup_epochs\n"
            "    else:\n        lr = base_lr\n"
            "    for param_group in optimizer.param_groups:\n"
            "        param_group['lr'] = lr\n```\n"
        ),
        "conference": "ICLR",
        "year": 2024,
    },
]

# Create rubrics
rubrics = {
    "demo_paper_001": {
        "name": "root",
        "weight": 1.0,
        "criteria": "",
        "children": [
            {
                "name": "core_implementation",
                "weight": 0.5,
                "criteria": "",
                "children": [
                    {"name": "channel_attention", "weight": 0.5, "criteria": "Implements channel attention module correctly"},
                    {"name": "spatial_attention", "weight": 0.3, "criteria": "Implements spatial attention module correctly"},
                    {"name": "feature_aggregation", "weight": 0.2, "criteria": "Combines both attention types for aggregation"},
                ],
            },
            {
                "name": "experiments",
                "weight": 0.3,
                "criteria": "",
                "children": [
                    {"name": "cifar10_setup", "weight": 0.5, "criteria": "Correct CIFAR-10 training setup"},
                    {"name": "results_reproduction", "weight": 0.5, "criteria": "Achieves close to reported accuracy"},
                ],
            },
            {"name": "code_quality", "weight": 0.2, "criteria": "Clean, well-documented code"},
        ],
    },
    "demo_paper_002": {
        "name": "root",
        "weight": 1.0,
        "criteria": "",
        "children": [
            {"name": "topk_implementation", "weight": 0.5, "criteria": "Correct TopK sparsification"},
            {"name": "error_feedback", "weight": 0.3, "criteria": "Error feedback mechanism works"},
            {"name": "compression_test", "weight": 0.2, "criteria": "Demonstrates compression ratio"},
        ],
    },
    "demo_paper_003": {
        "name": "root",
        "weight": 1.0,
        "criteria": "",
        "children": [
            {"name": "warmup_impl", "weight": 0.5, "criteria": "Correct linear warmup implementation"},
            {"name": "experiment_setup", "weight": 0.3, "criteria": "Proper training loop with warmup"},
            {"name": "results", "weight": 0.2, "criteria": "Compares warmup strategies"},
        ],
    },
}

# Create trajectories (simulated expert trajectories)
def make_trajectory(paper_id, paper):
    turns = [
        {"role": "assistant", "content": "Let me start by reading the paper to understand the method.",
         "tool_calls": [{"name": "read_paper", "arguments": {}}], "tool_results": [], "timestamp": 0},
        {"role": "user", "content": paper["markdown"][:2000],
         "tool_calls": [], "tool_results": [{"name": "read_paper", "output": "paper content"}], "timestamp": 1},
        {"role": "assistant", "content": "I've read the paper. Let me understand the key components. " + paper["markdown"][:500],
         "tool_calls": [{"name": "read_paper", "arguments": {"section": "method"}}], "tool_results": [], "timestamp": 2},
        {"role": "user", "content": "Method section: " + paper["markdown"][500:1500],
         "tool_calls": [], "tool_results": [{"name": "read_paper", "output": "method section"}], "timestamp": 3},
        {"role": "assistant", "content": "Now I understand the method. Let me start implementing.",
         "tool_calls": [{"name": "read_paper", "arguments": {"section": "code"}}], "tool_results": [], "timestamp": 4},
        {"role": "user", "content": "Code section with implementation details.",
         "tool_calls": [], "tool_results": [{"name": "read_paper", "output": "code"}], "timestamp": 5},
        {"role": "assistant", "content": "Let me set up the project structure first.",
         "tool_calls": [{"name": "bash", "arguments": {"command": "mkdir -p src && ls"}}], "tool_results": [], "timestamp": 6},
        {"role": "user", "content": "src/",
         "tool_calls": [], "tool_results": [{"name": "bash", "output": "src/"}], "timestamp": 7},
        {"role": "assistant", "content": "Now let me implement the core module based on the paper's method.",
         "tool_calls": [{"name": "write_file", "arguments": {"path": "src/model.py", "content": "import torch\nimport torch.nn as nn\n\nclass Model(nn.Module):\n    pass"}}],
         "tool_results": [], "timestamp": 8},
        {"role": "user", "content": "File written successfully.",
         "tool_calls": [], "tool_results": [{"name": "write_file", "output": "OK"}], "timestamp": 9},
        {"role": "assistant", "content": "Let me write the training script with the hyperparameters from the paper.",
         "tool_calls": [{"name": "write_file", "arguments": {"path": "src/train.py", "content": "import torch\n# training code"}}],
         "tool_results": [], "timestamp": 10},
        {"role": "user", "content": "File written successfully.",
         "tool_calls": [], "tool_results": [{"name": "write_file", "output": "OK"}], "timestamp": 11},
        {"role": "assistant", "content": "Now let me run the training to verify it works.",
         "tool_calls": [{"name": "bash", "arguments": {"command": "python src/train.py"}}],
         "tool_results": [], "timestamp": 12},
        {"role": "user", "content": "Training started...\nEpoch 1/10: loss=2.31\nEpoch 2/10: loss=1.85\nTraining complete.",
         "tool_calls": [], "tool_results": [{"name": "bash", "output": "Training complete."}], "timestamp": 13},
        {"role": "assistant", "content": "Training works. The implementation reproduces the paper's approach. Let me end the task.",
         "tool_calls": [{"name": "end_task", "arguments": {}}],
         "tool_results": [], "timestamp": 14},
        {"role": "user", "content": "Task ended.",
         "tool_calls": [], "tool_results": [{"name": "end_task", "output": "Task ended."}], "timestamp": 15},
    ]
    return {
        "task_id": f"{paper_id}_expert",
        "paper_id": paper_id,
        "total_time_seconds": 300,
        "final_score": 0.85,
        "turns": turns,
    }


if __name__ == "__main__":
    # Save papers
    papers_dir = Path("data/papers")
    papers_dir.mkdir(parents=True, exist_ok=True)
    with open(papers_dir / "papers.jsonl", "w") as f:
        for p in papers:
            f.write(json.dumps(p) + "\n")
    print(f"Created {len(papers)} papers in data/papers/papers.jsonl")

    # Save rubrics
    rubrics_dir = Path("data/rubrics")
    rubrics_dir.mkdir(parents=True, exist_ok=True)
    for pid, rubric in rubrics.items():
        with open(rubrics_dir / f"{pid}.json", "w") as f:
            json.dump(rubric, f, indent=2)
    print(f"Created {len(rubrics)} rubrics in data/rubrics/")

    # Save trajectories
    traj_dir = Path("data/trajectories")
    traj_dir.mkdir(parents=True, exist_ok=True)
    for p in papers:
        traj = make_trajectory(p["paper_id"], p)
        with open(traj_dir / f"{p['paper_id']}.json", "w") as f:
            json.dump(traj, f, indent=2)
    print(f"Created {len(papers)} trajectories in data/trajectories/")

    print("\nDemo data ready! You can now run the full pipeline.")
