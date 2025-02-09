from config import TrainingConfig, Metadata
from trainer import QwenChatTrainer
import logging
import argparse

def main():
    parser = argparse.ArgumentParser(description='Fine-tune Qwen model for chat')
    parser.add_argument('--train_data', type=str, required=True,
                      help='Path to training data JSON file')
    parser.add_argument('--eval_data', type=str, default=None,
                      help='Path to evaluation data JSON file')
    parser.add_argument('--model_name', type=str, default="Qwen-0.5B",
                      help='Name or path of the base model')
    args = parser.parse_args()

    # Initialize config and metadata
    config = TrainingConfig(model_name=args.model_name)
    metadata = Metadata()

    # Initialize and run trainer
    trainer = QwenChatTrainer(config, metadata)
    trainer.train(args.train_data, args.eval_data)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
