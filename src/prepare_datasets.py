import json 
from dataests import load_dataset
from typing import List, Dict


def convert_anthropic_to_chat_format(output_file: str, max_samples: int = 10000):
    """Convert Anthropic's Helpful-Harmless dataset to our chat format."""
    
    dataset = load_dataset("Anthropic/hh-rlhf")
    conversations = []
    
    for idx, item in enumerate(dataset['train']):
        if idx >= max_samples:
            break
            
        conversation = {
            "conversation_id": f"anthropic_{idx}",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant."
                },
                {
                    "role": "user",
                    "content": item['chosen'].split('Human: ')[1].split('Assistant: ')[0].strip()
                },
                {
                    "role": "assistant",
                    "content": item['chosen'].split('Assistant: ')[1].strip()
                }
            ],
            "metadata": {
                "language": "en",
                "timestamp": "2025-02-08 21:42:50",
                "user": "s3nh"
            }
        }
        conversations.append(conversation)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, indent=2)

# Usage
# convert_anthropic_to_chat_format('anthropic_chat.json')



def convert_sharegpt_to_chat_format(input_file: str, output_file: str):
    """Convert ShareGPT dataset to our chat format."""
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    conversations = []
    
    for idx, item in enumerate(data):
        messages = [{"role": "system", "content": "You are a helpful AI assistant."}]
        
        for turn in item['conversations']:
            role = "user" if turn['from'] == "human" else "assistant"
            messages.append({
                "role": role,
                "content": turn['value']
            })
            
        conversation = {
            "conversation_id": f"sharegpt_{idx}",
            "messages": messages,
            "metadata": {
                "language": "en",
                "timestamp": "2025-02-08 21:42:50",
                "user": "s3nh"
            }
        }
        conversations.append(conversation)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, indent=2)

# Usage
# convert_sharegpt_to_chat_format('sharegpt.json', 'sharegpt_chat.json')




def convert_oig_to_chat_format(output_file: str, max_samples: int = 10000):
    """Convert OIG dataset to our chat format."""
    
    dataset = load_dataset("laion/OIG")
    conversations = []
    
    for idx, item in enumerate(dataset['train']):
        if idx >= max_samples:
            break
            
        conversation = {
            "conversation_id": f"oig_{idx}",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant."
                },
                {
                    "role": "user",
                    "content": item['question']
                },
                {
                    "role": "assistant",
                    "content": item['answer']
                }
            ],
            "metadata": {
                "language": "en",
                "timestamp": "2025-02-08 21:42:50",
                "user": "s3nh",
                "source": "OIG"
            }
        }
        conversations.append(conversation)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, indent=2)

# Usage
# convert_oig_to_chat_format('oig_chat.json')


def combine_datasets(input_files: List[str], output_file: str):
    """Combine multiple datasets into one."""
    all_conversations = []
    
    for file in input_files:
        with open(file, 'r', encoding='utf-8') as f:
            conversations = json.load(f)
            all_conversations.extend(conversations)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_conversations, f, indent=2)

# Usage
# combine_datasets(
#     ['dolly_chat.json', 'anthropic_chat.json', 'sharegpt_chat.json', 'oig_chat.json'],
#     'combined_chat_dataset.json'
# )
