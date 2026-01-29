"""
Evaluation metrics for QA tasks.

Implements Exact Match (EM) and F1 score following standard QA evaluation.
"""

import re
import string
import logging
from collections import Counter
from typing import List, Tuple, Dict, Any

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluator for QA tasks with EM and F1 metrics."""
    
    @staticmethod
    def normalize_answer(text: str) -> str:
        """
        Normalize answer text for evaluation.
        
        Follows standard QA normalization:
        - Lowercase
        - Remove punctuation
        - Remove articles (a, an, the)
        - Remove extra whitespace
        
        Args:
            text: Answer text
            
        Returns:
            Normalized text
        """
        # Lowercase
        text = text.lower()
        
        # Remove punctuation
        text = ''.join(ch if ch not in string.punctuation else ' ' for ch in text)
        
        # Remove articles
        text = re.sub(r'\b(a|an|the)\b', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    @staticmethod
    def exact_match(prediction: str, ground_truth: str) -> float:
        """
        Compute exact match score.
        
        Args:
            prediction: Predicted answer
            ground_truth: Ground truth answer
            
        Returns:
            1.0 if exact match, 0.0 otherwise
        """
        pred_normalized = Evaluator.normalize_answer(prediction)
        gt_normalized = Evaluator.normalize_answer(ground_truth)
        
        return 1.0 if pred_normalized == gt_normalized else 0.0
    
    @staticmethod
    def f1_score(prediction: str, ground_truth: str) -> float:
        """
        Compute token-level F1 score.
        
        Args:
            prediction: Predicted answer
            ground_truth: Ground truth answer
            
        Returns:
            F1 score between 0 and 1
        """
        pred_tokens = Evaluator.normalize_answer(prediction).split()
        gt_tokens = Evaluator.normalize_answer(ground_truth).split()
        
        if not pred_tokens or not gt_tokens:
            return 0.0
        
        # Count common tokens
        pred_counter = Counter(pred_tokens)
        gt_counter = Counter(gt_tokens)
        
        common = pred_counter & gt_counter
        num_common = sum(common.values())
        
        if num_common == 0:
            return 0.0
        
        precision = num_common / len(pred_tokens)
        recall = num_common / len(gt_tokens)
        
        f1 = 2 * precision * recall / (precision + recall)
        
        return f1
    
    @staticmethod
    def evaluate_retrieval(
        retrieved_titles: List[str],
        supporting_facts: List[Tuple[str, int]],
    ) -> Dict[str, float]:
        """
        Evaluate retrieval quality using supporting facts.
        
        Args:
            retrieved_titles: List of retrieved document titles
            supporting_facts: List of (title, sentence_id) tuples
            
        Returns:
            Dictionary with retrieval metrics
        """
        if not supporting_facts:
            return {
                'retrieval_precision': 0.0,
                'retrieval_recall': 0.0,
                'retrieval_f1': 0.0,
            }
        
        # Extract gold titles
        gold_titles = set(title for title, _ in supporting_facts)
        retrieved_set = set(retrieved_titles)
        
        # Calculate metrics
        true_positives = len(gold_titles & retrieved_set)
        
        precision = true_positives / len(retrieved_set) if retrieved_set else 0.0
        recall = true_positives / len(gold_titles) if gold_titles else 0.0
        
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        return {
            'retrieval_precision': precision,
            'retrieval_recall': recall,
            'retrieval_f1': f1,
        }
    
    @staticmethod
    def compute_metrics(
        predictions: List[str],
        ground_truths: List[str],
        retrieved_docs_list: List[List[str]] = None,
        supporting_facts_list: List[List[Tuple[str, int]]] = None,
    ) -> Dict[str, Any]:
        """
        Compute all evaluation metrics.
        
        Args:
            predictions: List of predicted answers
            ground_truths: List of ground truth answers
            retrieved_docs_list: Optional list of retrieved document titles per question
            supporting_facts_list: Optional list of supporting facts per question
            
        Returns:
            Dictionary with all metrics
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("Number of predictions and ground truths must match")
        
        # Compute QA metrics
        em_scores = []
        f1_scores = []
        
        for pred, gt in zip(predictions, ground_truths):
            em_scores.append(Evaluator.exact_match(pred, gt))
            f1_scores.append(Evaluator.f1_score(pred, gt))
        
        metrics = {
            'exact_match': sum(em_scores) / len(em_scores) if em_scores else 0.0,
            'f1': sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
            'num_examples': len(predictions),
        }
        
        # Compute retrieval metrics if available
        if retrieved_docs_list and supporting_facts_list:
            retrieval_metrics = []
            
            for retrieved_docs, supporting_facts in zip(retrieved_docs_list, supporting_facts_list):
                if supporting_facts:
                    ret_metrics = Evaluator.evaluate_retrieval(retrieved_docs, supporting_facts)
                    retrieval_metrics.append(ret_metrics)
            
            if retrieval_metrics:
                metrics['retrieval_precision'] = sum(
                    m['retrieval_precision'] for m in retrieval_metrics
                ) / len(retrieval_metrics)
                metrics['retrieval_recall'] = sum(
                    m['retrieval_recall'] for m in retrieval_metrics
                ) / len(retrieval_metrics)
                metrics['retrieval_f1'] = sum(
                    m['retrieval_f1'] for m in retrieval_metrics
                ) / len(retrieval_metrics)
        
        return metrics
    
    @staticmethod
    def print_metrics(metrics: Dict[str, Any], prefix: str = ""):
        """
        Print metrics in a readable format.
        
        Args:
            metrics: Dictionary of metrics
            prefix: Optional prefix for logging
        """
        logger.info(f"{prefix}Evaluation Results:")
        logger.info(f"{prefix}  Exact Match: {metrics.get('exact_match', 0.0):.4f}")
        logger.info(f"{prefix}  F1 Score: {metrics.get('f1', 0.0):.4f}")
        logger.info(f"{prefix}  Num Examples: {metrics.get('num_examples', 0)}")
        
        if 'retrieval_precision' in metrics:
            logger.info(f"{prefix}  Retrieval Precision: {metrics['retrieval_precision']:.4f}")
            logger.info(f"{prefix}  Retrieval Recall: {metrics['retrieval_recall']:.4f}")
            logger.info(f"{prefix}  Retrieval F1: {metrics['retrieval_f1']:.4f}")
