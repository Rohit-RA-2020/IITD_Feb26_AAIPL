"""
Q-AGENT Fine-Tuning Script (Optimized)
======================================
Professional fine-tuning pipeline for Qwen2.5-14B with optimized hyperparameters,
comprehensive monitoring, and production-ready features.

Author: AAIPL Team
Model: Qwen2.5-14B-Instruct
Task: Question Agent Training
"""

import os

os.environ['HF_HOME'] = '/workspace/AAIPL/hf_cache'
os.environ['HF_DATASETS_CACHE'] = '/workspace/AAIPL/hf_cache/datasets'

import sys
from pathlib import Path
from datetime import datetime
import json
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel, train_on_responses_only
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback, EarlyStoppingCallback

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Centralized configuration for Q-Agent fine-tuning"""
    
    # Environment
    HF_HOME = '/workspace/AAIPL/hf_cache'
    HF_DATASETS_CACHE = '/workspace/AAIPL/hf_cache/datasets'
    
    # Model paths
    HF_CACHE_MODEL = Path("/workspace/AAIPL/hf_models/models--Qwen--Qwen2.5-14B-Instruct")
    
    # Data
    TRAIN_DATA_PATH = "data/question_agent/question_agent_train.json"
    VALIDATION_SPLIT = 0.15  # 15% for validation
    
    # Model configuration
    MAX_SEQ_LENGTH = 2048
    DTYPE = torch.bfloat16
    LOAD_IN_4BIT = True
    
    # LoRA configuration (optimized)
    LORA_R = 64  # Increased from 16 for better capacity
    LORA_ALPHA = 128  # 2x rank for stronger adaptation
    LORA_DROPOUT = 0.05  # Small dropout to prevent overfitting
    LORA_TARGET_MODULES = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "embed_tokens", "lm_head"  # Added for better adaptation
    ]
    
    # Training configuration (optimized)
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 8
    LEARNING_RATE = 1.5e-4  # Slightly lower for stability
    EPOCHS = 3
    WARMUP_RATIO = 0.1  # 10% warmup
    WEIGHT_DECAY = 0.01
    LR_SCHEDULER = "cosine"  # Cosine scheduling
    OPTIMIZER = "adamw_8bit"
    MAX_GRAD_NORM = 1.0  # Gradient clipping
    
    # Monitoring and saving
    LOGGING_STEPS = 5
    EVAL_STEPS = 25
    SAVE_STEPS = 50
    SAVE_TOTAL_LIMIT = 3  # Keep best 3 checkpoints
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 5
    EARLY_STOPPING_THRESHOLD = 0.001
    
    # Output paths
    OUTPUT_DIR = "outputs/question_agent_optimized"
    LORA_SAVE_PATH = "models/question_agent_lora_optimized"
    MERGED_SAVE_PATH = "models/question_agent_merged_optimized"
    
    # Misc
    SEED = 3407
    NUM_WORKERS = 0


# ============================================================================
# UTILITIES
# ============================================================================

class ColoredOutput:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text, char="=", color=ColoredOutput.HEADER):
    """Print a formatted header"""
    width = 80
    print(f"\n{color}{char * width}")
    print(f"{text.center(width)}")
    print(f"{char * width}{ColoredOutput.ENDC}\n")


def print_step(step_num, title, color=ColoredOutput.OKBLUE):
    """Print a formatted step header"""
    print(f"{color}{'─' * 80}")
    print(f"{'Step ' + str(step_num)}: {title}")
    print(f"{'─' * 80}{ColoredOutput.ENDC}")


def print_config_summary(config):
    """Print comprehensive configuration summary"""
    print_step("Config", "Configuration Summary", ColoredOutput.OKCYAN)
    
    categories = {
        "Model Configuration": {
            "Model": "Qwen2.5-14B-Instruct",
            "Agent Type": "Q-Agent (Question Generation)",
            "Max Sequence Length": config.MAX_SEQ_LENGTH,
            "Data Type": str(config.DTYPE).split('.')[-1],
            "Quantization": "4-bit" if config.LOAD_IN_4BIT else "None",
        },
        "LoRA Configuration": {
            "Rank (r)": config.LORA_R,
            "Alpha": config.LORA_ALPHA,
            "Dropout": config.LORA_DROPOUT,
            "Target Modules": len(config.LORA_TARGET_MODULES),
            "Trainable %": "~0.5-1.0%",
        },
        "Training Configuration": {
            "Batch Size": config.BATCH_SIZE,
            "Gradient Accumulation": config.GRADIENT_ACCUMULATION_STEPS,
            "Effective Batch Size": config.BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS,
            "Learning Rate": f"{config.LEARNING_RATE:.2e}",
            "Epochs": config.EPOCHS,
            "Warmup Ratio": f"{config.WARMUP_RATIO:.0%}",
            "LR Scheduler": config.LR_SCHEDULER,
            "Optimizer": config.OPTIMIZER,
            "Weight Decay": config.WEIGHT_DECAY,
        },
        "Monitoring & Checkpointing": {
            "Validation Split": f"{config.VALIDATION_SPLIT:.0%}",
            "Logging Steps": config.LOGGING_STEPS,
            "Evaluation Steps": config.EVAL_STEPS,
            "Save Steps": config.SAVE_STEPS,
            "Checkpoint Limit": config.SAVE_TOTAL_LIMIT,
            "Early Stopping": f"Yes (patience={config.EARLY_STOPPING_PATIENCE})",
        }
    }
    
    for category, items in categories.items():
        print(f"\n  {ColoredOutput.BOLD}{category}:{ColoredOutput.ENDC}")
        for key, value in items.items():
            print(f"    • {key}: {ColoredOutput.OKGREEN}{value}{ColoredOutput.ENDC}")


def save_training_config(config, output_dir):
    """Save configuration to JSON for reproducibility"""
    config_dict = {
        k: str(v) if isinstance(v, (Path, torch.dtype)) else v 
        for k, v in vars(config).items() 
        if not k.startswith('_') and k.isupper()
    }
    
    config_path = Path(output_dir) / "training_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"  ✓ Configuration saved to {config_path}")


class MetricsLogger(TrainerCallback):
    """Custom callback for enhanced training metrics logging"""
    
    def __init__(self):
        self.training_losses = []
        self.eval_losses = []
        self.learning_rates = []
        self.best_eval_loss = float('inf')
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            if 'loss' in logs:
                self.training_losses.append(logs['loss'])
            if 'eval_loss' in logs:
                self.eval_losses.append(logs['eval_loss'])
                if logs['eval_loss'] < self.best_eval_loss:
                    self.best_eval_loss = logs['eval_loss']
                    print(f"\n  🌟 New best validation loss: {self.best_eval_loss:.4f}")
            if 'learning_rate' in logs:
                self.learning_rates.append(logs['learning_rate'])


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def setup_environment():
    """Configure environment variables"""
    os.environ['HF_HOME'] = Config.HF_HOME
    os.environ['HF_DATASETS_CACHE'] = Config.HF_DATASETS_CACHE
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def load_model_and_tokenizer():
    """Load and configure the base model"""
    print_step(1, "Loading Qwen2.5-14B Model")
    
    # Resolve model path
    snapshot = (Config.HF_CACHE_MODEL / "refs" / "main").read_text().strip()
    model_path = str(Config.HF_CACHE_MODEL / "snapshots" / snapshot)
    
    print(f"  • Model Cache: {Config.HF_CACHE_MODEL.name}")
    print(f"  • Snapshot ID: {snapshot[:12]}...")
    print(f"  • Loading from: {model_path}")
    
    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=Config.MAX_SEQ_LENGTH,
        dtype=Config.DTYPE,
        load_in_4bit=Config.LOAD_IN_4BIT,
    )
    
    print(f"\n  {ColoredOutput.OKGREEN}✓ Model loaded successfully!{ColoredOutput.ENDC}")
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  • Total parameters: {total_params:,}")
    print(f"  • Model size: ~{total_params * 0.5 / 1e9:.1f}GB (4-bit)")
    
    return model, tokenizer


def configure_lora(model):
    """Add LoRA adapters to the model"""
    print_step(2, "Configuring LoRA Adapters")
    
    print(f"  • Rank (r): {Config.LORA_R}")
    print(f"  • Alpha: {Config.LORA_ALPHA}")
    print(f"  • Dropout: {Config.LORA_DROPOUT}")
    print(f"  • Target modules: {len(Config.LORA_TARGET_MODULES)}")
    print(f"    {', '.join(Config.LORA_TARGET_MODULES[:4])}...")
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=Config.LORA_R,
        target_modules=Config.LORA_TARGET_MODULES,
        lora_alpha=Config.LORA_ALPHA,
        lora_dropout=Config.LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=Config.SEED,
    )
    
    # Calculate trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_pct = 100 * trainable_params / total_params
    
    print(f"\n  {ColoredOutput.OKGREEN}✓ LoRA adapters configured!{ColoredOutput.ENDC}")
    print(f"  • Trainable parameters: {trainable_params:,}")
    print(f"  • Total parameters: {total_params:,}")
    print(f"  • Trainable percentage: {trainable_pct:.2f}%")
    
    return model


def setup_tokenizer(tokenizer):
    """Configure chat template and tokenizer settings"""
    print_step(3, "Setting Up Tokenizer & Chat Template")
    
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="chatml",
    )
    
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"  • Chat template: ChatML")
    print(f"  • Padding side: left")
    print(f"  • Pad token: {tokenizer.pad_token}")
    print(f"  • Vocab size: {len(tokenizer):,}")
    
    print(f"\n  {ColoredOutput.OKGREEN}✓ Tokenizer configured!{ColoredOutput.ENDC}")
    
    return tokenizer


def load_and_prepare_dataset(tokenizer):
    """Load and prepare training/validation datasets"""
    print_step(4, "Loading & Preparing Dataset")
    
    # Load dataset
    print(f"  • Loading from: {Config.TRAIN_DATA_PATH}")
    dataset = load_dataset(
        "json", 
        data_files=Config.TRAIN_DATA_PATH, 
        split="train"
    )
    
    print(f"  • Total examples: {len(dataset):,}")
    
    # Split into train/validation
    print(f"  • Creating validation split ({Config.VALIDATION_SPLIT:.0%})...")
    dataset = dataset.train_test_split(
        test_size=Config.VALIDATION_SPLIT, 
        seed=Config.SEED
    )
    
    print(f"  • Training examples: {len(dataset['train']):,}")
    print(f"  • Validation examples: {len(dataset['test']):,}")
    
    # Format dataset
    print(f"  • Applying chat template...")
    
    def formatting_prompts_func(examples):
        texts = []
        for convos in examples["conversations"]:
            text = tokenizer.apply_chat_template(
                convos, 
                tokenize=False, 
                add_generation_prompt=False
            )
            texts.append(text)
        return {"text": texts}
    
    dataset = dataset.map(
        formatting_prompts_func, 
        batched=True,
        desc="Formatting conversations"
    )
    
    # Calculate statistics
    total_tokens = sum(len(tokenizer(text)['input_ids']) for text in dataset['train']['text'][:100])
    avg_tokens = total_tokens / 100
    
    print(f"\n  {ColoredOutput.OKGREEN}✓ Dataset prepared!{ColoredOutput.ENDC}")
    print(f"  • Average sequence length: ~{avg_tokens:.0f} tokens (sampled)")
    print(f"  • Max sequence length: {Config.MAX_SEQ_LENGTH}")
    
    return dataset


def create_trainer(model, tokenizer, dataset):
    """Configure and create the SFT trainer"""
    print_step(5, "Configuring Training Pipeline")
    
    # Calculate training steps
    effective_batch_size = Config.BATCH_SIZE * Config.GRADIENT_ACCUMULATION_STEPS
    steps_per_epoch = len(dataset['train']) // effective_batch_size
    total_steps = steps_per_epoch * Config.EPOCHS
    warmup_steps = int(total_steps * Config.WARMUP_RATIO)
    
    print(f"  • Steps per epoch: {steps_per_epoch}")
    print(f"  • Total training steps: {total_steps}")
    print(f"  • Warmup steps: {warmup_steps}")
    print(f"  • Evaluation frequency: every {Config.EVAL_STEPS} steps")
    print(f"  • Checkpoint frequency: every {Config.SAVE_STEPS} steps")
    
    # Create metrics logger
    metrics_logger = MetricsLogger()
    
    # Configure trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        dataset_text_field="text",
        max_seq_length=Config.MAX_SEQ_LENGTH,
        args=SFTConfig(
            # Batch configuration
            per_device_train_batch_size=Config.BATCH_SIZE,
            per_device_eval_batch_size=Config.BATCH_SIZE,
            gradient_accumulation_steps=Config.GRADIENT_ACCUMULATION_STEPS,
            
            # Learning rate configuration
            learning_rate=Config.LEARNING_RATE,
            warmup_ratio=Config.WARMUP_RATIO,
            lr_scheduler_type=Config.LR_SCHEDULER,
            
            # Optimization
            optim=Config.OPTIMIZER,
            weight_decay=Config.WEIGHT_DECAY,
            max_grad_norm=Config.MAX_GRAD_NORM,
            
            # Training duration
            num_train_epochs=Config.EPOCHS,
            
            # Logging and evaluation
            logging_steps=Config.LOGGING_STEPS,
            eval_strategy="steps",
            eval_steps=Config.EVAL_STEPS,
            
            # Checkpointing
            save_strategy="steps",
            save_steps=Config.SAVE_STEPS,
            save_total_limit=Config.SAVE_TOTAL_LIMIT,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            
            # Output
            output_dir=Config.OUTPUT_DIR,
            
            # Hardware optimization
            bf16=True,
            gradient_checkpointing=True,
            dataloader_pin_memory=False,
            dataloader_num_workers=Config.NUM_WORKERS,
            
            # Misc
            seed=Config.SEED,
            remove_unused_columns=True,
            report_to="none",
            
            # Disable tqdm for cleaner output
            disable_tqdm=False,
        ),
        callbacks=[
            metrics_logger,
            EarlyStoppingCallback(
                early_stopping_patience=Config.EARLY_STOPPING_PATIENCE,
                early_stopping_threshold=Config.EARLY_STOPPING_THRESHOLD
            )
        ]
    )
    
    # Configure response-only training
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )
    
    print(f"\n  {ColoredOutput.OKGREEN}✓ Trainer configured!{ColoredOutput.ENDC}")
    print(f"  • Training only on assistant responses")
    print(f"  • Early stopping enabled (patience={Config.EARLY_STOPPING_PATIENCE})")
    
    return trainer, metrics_logger


def train_model(trainer, model):
    """Execute the training loop"""
    print_header("🚀 STARTING Q-AGENT TRAINING", "=", ColoredOutput.BOLD)
    
    print(f"  Training will take approximately 60-120 minutes")
    print(f"  Monitor the loss curves - both should decrease over time")
    print(f"  Best model will be saved automatically\n")
    
    start_time = datetime.now()
    print(f"  Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print("─" * 80)
    
    # Enable training mode
    FastLanguageModel.for_training(model)
    
    # Train
    try:
        trainer_stats = trainer.train()
        training_success = True
    except KeyboardInterrupt:
        print(f"\n\n{ColoredOutput.WARNING}⚠ Training interrupted by user{ColoredOutput.ENDC}")
        training_success = False
        trainer_stats = None
    except Exception as e:
        print(f"\n\n{ColoredOutput.FAIL}✗ Training failed: {e}{ColoredOutput.ENDC}")
        training_success = False
        trainer_stats = None
        raise
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "─" * 80)
    
    if training_success:
        print(f"\n{ColoredOutput.OKGREEN}✓ Training completed successfully!{ColoredOutput.ENDC}")
    
    print(f"  End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Duration: {str(duration).split('.')[0]}")
    
    return trainer_stats, training_success


def save_models(model, tokenizer, metrics_logger):
    """Save LoRA and merged models"""
    print_step(6, "Saving Trained Models")
    
    # Save LoRA adapters
    print(f"\n  📦 Saving LoRA adapters...")
    Path(Config.LORA_SAVE_PATH).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(Config.LORA_SAVE_PATH)
    tokenizer.save_pretrained(Config.LORA_SAVE_PATH)
    print(f"  {ColoredOutput.OKGREEN}✓{ColoredOutput.ENDC} LoRA saved: {Config.LORA_SAVE_PATH}")
    
    # Save merged model
    print(f"\n  🔄 Merging and saving full model...")
    print(f"     (This may take 5-10 minutes)")
    Path(Config.MERGED_SAVE_PATH).mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(
        Config.MERGED_SAVE_PATH, 
        tokenizer, 
        save_method="merged_16bit"
    )
    print(f"  {ColoredOutput.OKGREEN}✓{ColoredOutput.ENDC} Merged model saved: {Config.MERGED_SAVE_PATH}")
    
    # Save training metrics
    metrics_path = Path(Config.OUTPUT_DIR) / "training_metrics.json"
    metrics = {
        "best_eval_loss": metrics_logger.best_eval_loss,
        "final_training_losses": metrics_logger.training_losses[-10:] if metrics_logger.training_losses else [],
        "final_eval_losses": metrics_logger.eval_losses[-5:] if metrics_logger.eval_losses else [],
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  {ColoredOutput.OKGREEN}✓{ColoredOutput.ENDC} Metrics saved: {metrics_path}")


def print_final_summary(dataset, trainer_stats, metrics_logger, training_success):
    """Print comprehensive training summary"""
    print_header("📊 Q-AGENT TRAINING SUMMARY", "=", ColoredOutput.OKCYAN)
    
    print(f"  {ColoredOutput.BOLD}Dataset Statistics:{ColoredOutput.ENDC}")
    print(f"    • Training examples: {len(dataset['train']):,}")
    print(f"    • Validation examples: {len(dataset['test']):,}")
    print(f"    • Total epochs: {Config.EPOCHS}")
    
    if training_success and trainer_stats:
        print(f"\n  {ColoredOutput.BOLD}Training Results:{ColoredOutput.ENDC}")
        print(f"    • Final training loss: {ColoredOutput.OKGREEN}{trainer_stats.training_loss:.4f}{ColoredOutput.ENDC}")
        if metrics_logger.best_eval_loss != float('inf'):
            print(f"    • Best validation loss: {ColoredOutput.OKGREEN}{metrics_logger.best_eval_loss:.4f}{ColoredOutput.ENDC}")
        print(f"    • Total training steps: {trainer_stats.global_step:,}")
        
        # Loss improvement
        if len(metrics_logger.training_losses) > 1:
            initial_loss = metrics_logger.training_losses[0]
            final_loss = metrics_logger.training_losses[-1]
            improvement = ((initial_loss - final_loss) / initial_loss) * 100
            print(f"    • Training loss improvement: {ColoredOutput.OKGREEN}{improvement:.1f}%{ColoredOutput.ENDC}")
    
    print(f"\n  {ColoredOutput.BOLD}Model Outputs:{ColoredOutput.ENDC}")
    print(f"    • LoRA adapters: {Config.LORA_SAVE_PATH}")
    print(f"    • Merged model: {Config.MERGED_SAVE_PATH}")
    print(f"    • Training logs: {Config.OUTPUT_DIR}")
    print(f"    • Configuration: {Config.OUTPUT_DIR}/training_config.json")
    
    print(f"\n  {ColoredOutput.BOLD}Model Details:{ColoredOutput.ENDC}")
    print(f"    • Base model: Qwen2.5-14B-Instruct")
    print(f"    • Agent type: Q-Agent (Question Generation)")
    print(f"    • LoRA rank: {Config.LORA_R}")
    print(f"    • LoRA alpha: {Config.LORA_ALPHA}")
    print(f"    • Learning rate: {Config.LEARNING_RATE:.2e}")
    print(f"    • Scheduler: {Config.LR_SCHEDULER}")


def main():
    """Main training pipeline"""
    try:
        # Header
        print_header("🤖 Q-AGENT FINE-TUNING (OPTIMIZED)", "=")
        print(f"  Model: Qwen2.5-14B-Instruct")
        print(f"  Task: Question Agent Training")
        print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Setup
        setup_environment()
        print_config_summary(Config)
        save_training_config(Config, Config.OUTPUT_DIR)
        
        # Pipeline
        model, tokenizer = load_model_and_tokenizer()
        model = configure_lora(model)
        tokenizer = setup_tokenizer(tokenizer)
        dataset = load_and_prepare_dataset(tokenizer)
        trainer, metrics_logger = create_trainer(model, tokenizer, dataset)
        trainer_stats, training_success = train_model(trainer, model)
        
        if training_success:
            save_models(model, tokenizer, metrics_logger)
        
        # Summary
        print_final_summary(dataset, trainer_stats, metrics_logger, training_success)
        
        # Final message
        print_header("✨ Q-AGENT TRAINING PIPELINE COMPLETE", "=", ColoredOutput.OKGREEN)
        
        if training_success:
            print(f"  {ColoredOutput.BOLD}Next Steps:{ColoredOutput.ENDC}")
            print(f"    1. Test your Q-Agent model with inference script")
            print(f"    2. Evaluate question generation quality")
            print(f"    3. Compare with baseline model")
            print(f"    4. Integrate with A-Agent for complete system")
            print(f"\n  Your optimized Q-Agent is ready! 🎉\n")
        else:
            print(f"  {ColoredOutput.WARNING}Training did not complete successfully.{ColoredOutput.ENDC}")
            print(f"  Check the logs above for details.\n")
        
        return 0 if training_success else 1
        
    except Exception as e:
        print(f"\n{ColoredOutput.FAIL}{'=' * 80}")
        print(f"✗ FATAL ERROR")
        print(f"{'=' * 80}{ColoredOutput.ENDC}\n")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        print(f"\n  Traceback:")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)