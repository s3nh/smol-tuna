import logging
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from data_processor import ConversationDataProcessor
from config import TrainingConfig, Metadata
import wandb
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Custom loss function.
        """
        labels = inputs.get("labels")
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        loss_fct = torch.nn.CrossEntropyLoss()
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

class QwenChatTrainer:
    def __init__(self, config: TrainingConfig, metadata: Metadata):
        self.config = config
        self.metadata = metadata
        self.setup_logging()
        
    def setup_logging(self):
        """Initialize logging and wandb."""
        wandb.init(
            project="qwen-chat-finetuning",
            name=self.metadata.experiment_name,
            config={
                **self.config.__dict__,
                "timestamp": self.metadata.timestamp,
                "user": self.metadata.user
            }
        )
        
    def load_model_and_tokenizer(self):
        """Load the pre-trained model and tokenizer."""
        logger.info(f"Loading model and tokenizer: {self.config.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        
        # Add special tokens for chat
        special_tokens = {
            "additional_special_tokens": [
                "<|system|>", "<|user|>", "<|assistant|>", "</s>"
            ]
        }
        tokenizer.add_special_tokens(special_tokens)
        model.resize_token_embeddings(len(tokenizer))
        
        return model, tokenizer
        
    def train(self, train_data_path: str, eval_data_path: str = None):
        """Main training function."""
        model, tokenizer = self.load_model_and_tokenizer()
        
        # Initialize data processor
        processor = ConversationDataProcessor(tokenizer, self.config.max_length)
        
        # Load and process datasets
        logger.info("Loading and processing training data")
        train_dataset = processor.load_conversations(train_data_path)
        train_dataset = processor.tokenize_conversations(train_dataset)
        
        if eval_data_path:
            eval_dataset = processor.load_conversations(eval_data_path)
            eval_dataset = processor.tokenize_conversations(eval_dataset)
        else:
            eval_dataset = None
            
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            logging_dir=os.path.join(self.config.output_dir, 'logs'),
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            save_total_limit=self.config.save_total_limit,
            evaluation_strategy="steps" if eval_dataset else "no",
            report_to="wandb",
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False
            )
        )
        
        # Start training
        logger.info("Starting training")
        trainer.train()
        
        # Save the final model
        logger.info(f"Saving model to {self.config.output_dir}")
        trainer.save_model()
        tokenizer.save_pretrained(self.config.output_dir)
