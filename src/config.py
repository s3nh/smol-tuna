from dataclasses import dataclass
from datetime import datetime

@dataclass
class TrainingConfig:
    model_name: str = "Qwen-0.5B"
    output_dir: str = "fine_tuned_qwen_chat"
    max_length: int = 512
    batch_size: int = 4
    num_epochs: int = 3
    learning_rate: float = 2e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 2

@dataclass
class Metadata:
    timestamp: str = "2025-02-08 21:32:32"
    user: str = "s3nh"
    experiment_name: str = f"qwen_chat_ft_{datetime.now().strftime('%Y%m%d_%H%M%S')}"



