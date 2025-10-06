import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import json
import time
from tqdm import tqdm
import sys

# Add src to path
sys.path.append('src')

from data_loaders.clevrDataset import CLEVRDataset, clevrDataloader
from models.neuro_symbolic_vqa import NeuroSymbolicVQA, VQALoss, build_model
from models.reasoning.question_encoder import QuestionVocabulary, ProgramVocabulary
from utils.config import Config, get_default_config, get_debug_config


class Trainer:
    """
    Trainer class for Neuro-Symbolic VQA
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.training.device if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
        
        # Create directories
        config.paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        config.paths.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize vocabularies
        print("\nBuilding vocabularies...")
        self.question_vocab, self.program_vocab, self.answer_vocab = self._build_vocabularies()
        
        # Save vocabularies
        self.question_vocab.save(config.paths.root_dir / 'data' / 'question_vocab.json')
        vocab_path = config.paths.root_dir / 'data' / 'vocab_info.json'
        with open(vocab_path, 'w') as f:
            json.dump({
                'question_vocab_size': len(self.question_vocab),
                'program_vocab_size': len(self.program_vocab),
                'answer_vocab_size': len(self.answer_vocab),
                'answer_to_idx': self.answer_vocab
            }, f, indent=2)
        
        print(f"Question vocab size: {len(self.question_vocab)}")
        print(f"Program vocab size: {len(self.program_vocab)}")
        print(f"Answer vocab size: {len(self.answer_vocab)}")
        
        # Build dataloaders
        print("\nLoading datasets...")
        self.train_loader = self._build_dataloader('train')
        self.val_loader = self._build_dataloader('val')
        
        # Build model
        print("\nBuilding model...")
        self.model = build_model(config, self.question_vocab, self.program_vocab, self.answer_vocab)
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Loss function
        self.criterion = VQALoss(
            attribute_weight=0.5,
            program_weight=config.training.program_loss_weight,
            answer_weight=config.training.answer_loss_weight
        )
        
        # Optimizer
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        
        # Tensorboard
        self.writer = SummaryWriter(log_dir=config.paths.log_dir / config.experiment_name)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_acc = 0.0
        
    def _build_vocabularies(self):
        """Build vocabularies from training data"""
        # Load training questions to build vocab
        train_dataset = CLEVRDataset(
            data_dir=str(self.config.data.data_dir),
            split='train',
            max_samples=self.config.data.max_train_samples
        )
        
        # Question vocabulary
        question_vocab = QuestionVocabulary()
        questions = [q['question'] for q in train_dataset.questions]
        question_vocab.build_from_questions(questions, min_count=1)
        
        # Program vocabulary
        program_vocab = ProgramVocabulary()
        
        # Answer vocabulary
        answer_vocab = train_dataset.answer_to_idx
        
        return question_vocab, program_vocab, answer_vocab
    
    def _build_dataloader(self, split: str) -> DataLoader:
        """Build dataloader for given split"""
        max_samples = None
        if split == 'train' and self.config.data.max_train_samples:
            max_samples = self.config.data.max_train_samples
        elif split == 'val' and self.config.data.max_val_samples:
            max_samples = self.config.data.max_val_samples
        
        return clevrDataloader(
            data_dir=str(self.config.data.data_dir),
            split=split,
            batch_size=self.config.data.batch_size,
            shuffle=(split == 'train'),
            num_workers=self.config.data.num_workers,
            max_samples=max_samples
        )
    
    def _build_optimizer(self):
        """Build optimizer"""
        if self.config.training.optimizer == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        else:
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                momentum=0.9,
                weight_decay=self.config.training.weight_decay
            )
    
    def _build_scheduler(self):
        """Build learning rate scheduler"""
        if self.config.training.scheduler == 'reduce_on_plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=self.config.training.factor,
                patience=self.config.training.patience,
                min_lr=self.config.training.min_lr
            )
        elif self.config.training.scheduler == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.num_epochs
            )
        else:
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.5
            )
    
    def _prepare_batch(self, batch):
        """Prepare batch for training"""
        images = batch['images'].to(self.device)
        
        # Encode questions
        question_indices = []
        question_lengths = []
        for question in batch['questions']:
            indices = self.question_vocab.encode(question)
            question_indices.append(indices)
            question_lengths.append(len(indices))
        
        # Pad questions
        max_len = max(question_lengths)
        padded_questions = torch.zeros(len(question_indices), max_len, dtype=torch.long)
        for i, indices in enumerate(question_indices):
            padded_questions[i, :len(indices)] = torch.tensor(indices)
        
        padded_questions = padded_questions.to(self.device)
        question_lengths = torch.tensor(question_lengths).to(self.device)
        
        # Answer targets
        answer_targets = batch['answer_indices'].to(self.device)
        
        # Program targets (if available)
        program_targets = None
        if 'programs' in batch:
            program_targets = self._encode_programs(batch['programs'])
            program_targets = program_targets.to(self.device)
        
        return {
            'images': images,
            'question_indices': padded_questions,
            'question_lengths': question_lengths,
            'answer_targets': answer_targets,
            'program_targets': program_targets
        }
    
    def _encode_programs(self, programs):
        """Encode programs to indices"""
        encoded_programs = []
        max_len = 27  # Max program length
        
        for program in programs:
            indices = self.program_vocab.encode_program(program)
            encoded_programs.append(indices)
        
        # Pad programs
        padded = torch.zeros(len(encoded_programs), max_len, dtype=torch.long)
        for i, indices in enumerate(encoded_programs):
            length = min(len(indices), max_len)
            padded[i, :length] = torch.tensor(indices[:length])
        
        return padded
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Prepare batch
            prepared_batch = self._prepare_batch(batch)
            
            # Forward pass
            outputs = self.model(
                images=prepared_batch['images'],
                question_indices=prepared_batch['question_indices'],
                question_lengths=prepared_batch['question_lengths'],
                program_indices=prepared_batch['program_targets']
            )
            
            # Compute loss
            targets = {
                'answer_targets': prepared_batch['answer_targets']
            }
            if prepared_batch['program_targets'] is not None:
                targets['program_targets'] = prepared_batch['program_targets']
            
            losses = self.criterion(outputs, targets)
            loss = losses['total_loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.training.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.grad_clip
                )
            
            self.optimizer.step()
            
            # Compute accuracy
            predictions = outputs['answer_logits'].argmax(dim=1)
            correct += (predictions == prepared_batch['answer_targets']).sum().item()
            total += prepared_batch['answer_targets'].size(0)
            
            # Update metrics
            total_loss += loss.item()
            
            # Logging
            if batch_idx % self.config.training.log_every == 0:
                avg_loss = total_loss / (batch_idx + 1)
                accuracy = 100.0 * correct / total if total > 0 else 0
                
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'acc': f'{accuracy:.2f}%'
                })
                
                # Tensorboard
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/accuracy', accuracy, self.global_step)
                
                for key, value in losses.items():
                    if key != 'total_loss':
                        self.writer.add_scalar(f'train/{key}', value.item(), self.global_step)
            
            self.global_step += 1
        
        return total_loss / len(self.train_loader), 100.0 * correct / total
    
    def validate(self):
        """Validate on validation set"""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                # Prepare batch
                prepared_batch = self._prepare_batch(batch)
                
                # Forward pass
                outputs = self.model(
                    images=prepared_batch['images'],
                    question_indices=prepared_batch['question_indices'],
                    question_lengths=prepared_batch['question_lengths'],
                    program_indices=prepared_batch['program_targets']
                )
                
                # Compute loss
                targets = {
                    'answer_targets': prepared_batch['answer_targets']
                }
                if prepared_batch['program_targets'] is not None:
                    targets['program_targets'] = prepared_batch['program_targets']
                
                losses = self.criterion(outputs, targets)
                loss = losses['total_loss']
                
                # Compute accuracy
                predictions = outputs['answer_logits'].argmax(dim=1)
                correct += (predictions == prepared_batch['answer_targets']).sum().item()
                total += prepared_batch['answer_targets'].size(0)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = self.config.paths.checkpoint_dir / f'checkpoint_epoch_{self.current_epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = self.config.paths.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"Saved best model: {best_path}")
    
    def train(self):
        """Main training loop"""
        print("\n" + "=" * 80)
        print("STARTING TRAINING")
        print("=" * 80)
        
        for epoch in range(self.config.training.num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            print(f"\nEpoch {epoch}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            
            # Validate
            if epoch % self.config.training.validate_every == 0:
                val_loss, val_acc = self.validate()
                print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                
                # Tensorboard
                self.writer.add_scalar('val/loss', val_loss, epoch)
                self.writer.add_scalar('val/accuracy', val_acc, epoch)
                
                # Update scheduler
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_acc)
                else:
                    self.scheduler.step()
                
                # Save checkpoint
                is_best = val_acc > self.best_val_acc
                if is_best:
                    self.best_val_acc = val_acc
                    print(f"  New best validation accuracy: {val_acc:.2f}%")
                
                if epoch % self.config.training.save_every == 0 or is_best:
                    self.save_checkpoint(is_best=is_best)
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        print("=" * 80)
        
        self.writer.close()


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Neuro-Symbolic VQA')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--debug', action='store_true', help='Use debug configuration')
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = Config.load(args.config)
    elif args.debug:
        config = get_debug_config()
    else:
        config = get_default_config()
    
    # Print configuration
    config.print_config()
    
    # Set random seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    
    # Create trainer
    trainer = Trainer(config)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()