import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import json
from tqdm import tqdm
from collections import defaultdict
import sys

# Add src to path
sys.path.append('src')

from data_loaders.clevrDataset import CLEVRDataset, clevrDataloader
from models.neuro_symbolic_vqa import NeuroSymbolicVQA
from models.reasoning.question_encoder import QuestionVocabulary, ProgramVocabulary
from utils.config import Config


class Evaluator:
    """
    Evaluator for Neuro-Symbolic VQA
    """
    
    def __init__(self, checkpoint_path: str, config_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load config
        if config_path:
            self.config = Config.load(config_path)
        else:
            self.config = checkpoint['config']
        
        # Load vocabularies
        print("Loading vocabularies...")
        self.question_vocab = QuestionVocabulary.load('data/question_vocab.json')
        self.program_vocab = ProgramVocabulary()
        
        with open('data/vocab_info.json', 'r') as f:
            vocab_info = json.load(f)
        
        self.answer_vocab = vocab_info['answer_to_idx']
        self.idx_to_answer = {v: k for k, v in self.answer_vocab.items()}
        
        # Build model
        print("Building model...")
        self.model = NeuroSymbolicVQA(
            question_vocab_size=len(self.question_vocab),
            program_vocab_size=len(self.program_vocab),
            answer_vocab_size=len(self.answer_vocab),
            visual_feature_dim=self.config.visual.num_object_features,
            question_embedding_dim=self.config.question.embedding_dim,
            question_hidden_dim=self.config.question.hidden_dim,
            max_objects=self.config.visual.max_objects_per_image,
            device=str(self.device)
        )
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print("Model loaded successfully!")
        print(f"Checkpoint from epoch: {checkpoint['epoch']}")
        print(f"Best validation accuracy: {checkpoint['best_val_acc']:.2f}%")
    
    def evaluate(self, split: str = 'val', max_samples: int = None):
        """
        Evaluate model on dataset
        
        Args:
            split: Dataset split ('val' or 'test')
            max_samples: Maximum samples to evaluate (None for all)
        
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\nEvaluating on {split} split...")
        
        # Load dataset
        dataloader = clevrDataloader(
            data_dir=str(self.config.data.data_dir),
            split=split,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            max_samples=max_samples
        )
        
        # Evaluation metrics
        total_correct = 0
        total_samples = 0
        
        # Per-question-type metrics
        type_correct = defaultdict(int)
        type_total = defaultdict(int)
        
        # Per-answer metrics
        answer_correct = defaultdict(int)
        answer_total = defaultdict(int)
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Prepare batch
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
                
                # Forward pass
                outputs = self.model(
                    images=images,
                    question_indices=padded_questions,
                    question_lengths=question_lengths
                )
                
                # Get predictions
                predictions = outputs['answer_logits'].argmax(dim=1)
                targets = batch['answer_indices'].to(self.device)
                
                # Compute accuracy
                correct = (predictions == targets)
                total_correct += correct.sum().item()
                total_samples += targets.size(0)
                
                # Per-answer accuracy
                for pred, target, answer in zip(predictions, targets, batch['answers']):
                    answer_total[answer] += 1
                    if pred == target:
                        answer_correct[answer] += 1
                
                # Per-question-type accuracy (if programs available)
                if 'programs' in batch:
                    for i, program in enumerate(batch['programs']):
                        q_type = program[-1]['type']
                        type_total[q_type] += 1
                        if correct[i]:
                            type_correct[q_type] += 1
        
        # Compute overall accuracy
        overall_accuracy = 100.0 * total_correct / total_samples
        
        # Compute per-question-type accuracy
        type_accuracies = {}
        for q_type in type_total:
            if type_total[q_type] > 0:
                type_accuracies[q_type] = 100.0 * type_correct[q_type] / type_total[q_type]
        
        # Compute per-answer accuracy
        answer_accuracies = {}
        for answer in answer_total:
            if answer_total[answer] > 0:
                answer_accuracies[answer] = 100.0 * answer_correct[answer] / answer_total[answer]
        
        # Compile results
        results = {
            'overall_accuracy': overall_accuracy,
            'total_samples': total_samples,
            'correct_samples': total_correct,
            'question_type_accuracy': type_accuracies,
            'answer_accuracy': answer_accuracies
        }
        
        return results
    
    def print_results(self, results: dict):
        """Print evaluation results"""
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)
        
        print(f"\nOverall Accuracy: {results['overall_accuracy']:.2f}%")
        print(f"Total Samples: {results['total_samples']}")
        print(f"Correct Predictions: {results['correct_samples']}")
        
        # Question type accuracies
        if results['question_type_accuracy']:
            print("\n" + "-" * 80)
            print("Per-Question-Type Accuracy:")
            print("-" * 80)
            
            sorted_types = sorted(
                results['question_type_accuracy'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            for q_type, accuracy in sorted_types:
                print(f"  {q_type:25s}: {accuracy:6.2f}%")
        
        # Top answers
        if results['answer_accuracy']:
            print("\n" + "-" * 80)
            print("Top Answer Accuracies (showing top 20):")
            print("-" * 80)
            
            sorted_answers = sorted(
                results['answer_accuracy'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:20]
            
            for answer, accuracy in sorted_answers:
                print(f"  {answer:20s}: {accuracy:6.2f}%")
        
        print("\n" + "=" * 80)
    
    def save_results(self, results: dict, output_path: str):
        """Save results to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {output_path}")
    
    def evaluate_single_image(self, image_path: str, question: str):
        """
        Evaluate on a single image and question
        
        Args:
            image_path: Path to image
            question: Question string
        
        Returns:
            answer: Predicted answer
        """
        from PIL import Image
        import torchvision.transforms as transforms
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image = image.resize((320, 240))
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # Encode question
        question_indices = self.question_vocab.encode(question)
        question_length = len(question_indices)
        
        padded_question = torch.zeros(1, question_length, dtype=torch.long)
        padded_question[0, :question_length] = torch.tensor(question_indices)
        padded_question = padded_question.to(self.device)
        
        question_length = torch.tensor([question_length]).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(
                images=image_tensor,
                question_indices=padded_question,
                question_lengths=question_length
            )
        
        # Get prediction
        prediction_idx = outputs['answer_logits'][0].argmax().item()
        answer = self.idx_to_answer.get(prediction_idx, "unknown")
        
        return answer


def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Neuro-Symbolic VQA')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'], help='Dataset split')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum samples to evaluate')
    parser.add_argument('--output', type=str, default='outputs/eval_results.json', help='Output path for results')
    parser.add_argument('--single_image', type=str, default=None, help='Path to single image for testing')
    parser.add_argument('--question', type=str, default=None, help='Question for single image')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = Evaluator(args.checkpoint, args.config)
    
    if args.single_image and args.question:
        # Evaluate single image
        print(f"\nEvaluating single image...")
        print(f"Image: {args.single_image}")
        print(f"Question: {args.question}")
        
        answer = evaluator.evaluate_single_image(args.single_image, args.question)
        print(f"Answer: {answer}")
    
    else:
        # Evaluate on dataset
        results = evaluator.evaluate(args.split, args.max_samples)
        
        # Print results
        evaluator.print_results(results)
        
        # Save results
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        evaluator.save_results(results, args.output)


if __name__ == "__main__":
    main()