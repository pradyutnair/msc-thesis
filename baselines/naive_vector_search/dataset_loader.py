"""
Dataset loader for multi-hop QA datasets.

Supports: 2WikiMultihopQA, HotpotQA, TriviaQA, Natural Questions
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a document with title and content."""
    id: str
    title: str
    text: str
    metadata: Optional[Dict] = None


@dataclass
class Question:
    """Represents a question with answer and supporting documents."""
    id: str
    question: str
    answer: str
    supporting_facts: Optional[List[Tuple[str, int]]] = None
    type: Optional[str] = None
    level: Optional[str] = None


class DatasetLoader:
    """Unified loader for multi-hop QA datasets."""
    
    SUPPORTED_DATASETS = {
        '2wikimultihopqa': '2wikimultihopqa',
        'hotpotqa': 'hotpotqa',
        'triviaqa': 'triviaqa',
        'natural_questions': 'natural_questions',
        'nq': 'natural_questions',  # alias
    }
    
    def __init__(self, dataset_name: str, dataset_dir: Path):
        """
        Initialize dataset loader.
        
        Args:
            dataset_name: Name of the dataset
            dataset_dir: Path to dataset directory
        """
        self.dataset_name = self._normalize_dataset_name(dataset_name)
        self.dataset_dir = Path(dataset_dir)
        
        if not self.dataset_dir.exists():
            raise ValueError(f"Dataset directory not found: {self.dataset_dir}")
        
        logger.info(f"Initialized {self.dataset_name} loader from {self.dataset_dir}")
    
    def _normalize_dataset_name(self, name: str) -> str:
        """Normalize dataset name to standard format."""
        name_lower = name.lower()
        if name_lower not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Unsupported dataset: {name}. "
                f"Supported: {list(self.SUPPORTED_DATASETS.keys())}"
            )
        return self.SUPPORTED_DATASETS[name_lower]
    
    def load_questions(self, split: str = 'dev', limit: Optional[int] = None) -> List[Question]:
        """
        Load questions from dataset.
        
        Args:
            split: Dataset split (train/dev/test)
            limit: Maximum number of questions to load
            
        Returns:
            List of Question objects
        """
        logger.info(f"Loading {split} questions from {self.dataset_name}")
        
        if self.dataset_name == '2wikimultihopqa':
            return self._load_2wikimultihopqa(split, limit)
        elif self.dataset_name == 'hotpotqa':
            return self._load_hotpotqa(split, limit)
        elif self.dataset_name == 'triviaqa':
            return self._load_triviaqa(split, limit)
        elif self.dataset_name == 'natural_questions':
            return self._load_natural_questions(split, limit)
        else:
            raise NotImplementedError(f"Loader not implemented for {self.dataset_name}")
    
    def load_corpus(self) -> List[Document]:
        """
        Load document corpus for retrieval.
        
        Returns:
            List of Document objects
        """
        logger.info(f"Loading corpus from {self.dataset_name}")
        
        if self.dataset_name == '2wikimultihopqa':
            return self._load_2wikimultihopqa_corpus()
        elif self.dataset_name == 'hotpotqa':
            return self._load_hotpotqa_corpus()
        elif self.dataset_name == 'triviaqa':
            return self._load_triviaqa_corpus()
        elif self.dataset_name == 'natural_questions':
            return self._load_natural_questions_corpus()
        else:
            raise NotImplementedError(f"Corpus loader not implemented for {self.dataset_name}")
    
    def _load_2wikimultihopqa(self, split: str, limit: Optional[int]) -> List[Question]:
        """Load 2WikiMultihopQA questions."""
        file_path = self.dataset_dir / f"{split}.json"
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        questions = []
        for idx, item in enumerate(data[:limit] if limit else data):
            question = Question(
                id=item.get('_id', str(idx)),
                question=item['question'],
                answer=item['answer'],
                supporting_facts=item.get('supporting_facts', []),
                type=item.get('type'),
                level=item.get('level')
            )
            questions.append(question)
        
        logger.info(f"Loaded {len(questions)} questions from 2WikiMultihopQA {split}")
        return questions
    
    def _load_2wikimultihopqa_corpus(self) -> List[Document]:
        """Load 2WikiMultihopQA corpus from context."""
        # For 2WikiMultihopQA, we extract corpus from the dev/train files
        corpus_file = self.dataset_dir / "dev.json"
        
        if not corpus_file.exists():
            corpus_file = self.dataset_dir / "train.json"
        
        if not corpus_file.exists():
            raise FileNotFoundError(f"No corpus file found in {self.dataset_dir}")
        
        with open(corpus_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        doc_id = 0
        seen_titles = set()
        
        for item in data:
            contexts = item.get('context', [])
            for context in contexts:
                if isinstance(context, list) and len(context) >= 2:
                    title = context[0]
                    sentences = context[1]
                    
                    # Avoid duplicates
                    if title in seen_titles:
                        continue
                    seen_titles.add(title)
                    
                    # Combine sentences into text
                    if isinstance(sentences, list):
                        text = ' '.join(sentences)
                    else:
                        text = str(sentences)
                    
                    doc = Document(
                        id=str(doc_id),
                        title=title,
                        text=text,
                        metadata={'dataset': '2wikimultihopqa'}
                    )
                    documents.append(doc)
                    doc_id += 1
        
        logger.info(f"Loaded {len(documents)} documents from 2WikiMultihopQA corpus")
        return documents
    
    def _load_hotpotqa(self, split: str, limit: Optional[int]) -> List[Question]:
        """Load HotpotQA questions."""
        # HotpotQA files are named like: hotpot_dev_distractor_v1.json
        possible_files = [
            self.dataset_dir / f"hotpot_{split}_distractor_v1.json",
            self.dataset_dir / f"hotpot_{split}_fullwiki_v1.json",
            self.dataset_dir / f"{split}.json",
        ]
        
        file_path = None
        for path in possible_files:
            if path.exists():
                file_path = path
                break
        
        if not file_path:
            raise FileNotFoundError(
                f"No HotpotQA file found for split '{split}' in {self.dataset_dir}"
            )
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        questions = []
        for idx, item in enumerate(data[:limit] if limit else data):
            question = Question(
                id=item.get('_id', str(idx)),
                question=item['question'],
                answer=item['answer'],
                supporting_facts=item.get('supporting_facts', []),
                type=item.get('type'),
                level=item.get('level')
            )
            questions.append(question)
        
        logger.info(f"Loaded {len(questions)} questions from HotpotQA {split}")
        return questions
    
    def _load_hotpotqa_corpus(self) -> List[Document]:
        """Load HotpotQA corpus from context."""
        possible_files = [
            self.dataset_dir / "hotpot_dev_distractor_v1.json",
            self.dataset_dir / "hotpot_train_v1.1.json",
            self.dataset_dir / "dev.json",
        ]
        
        corpus_file = None
        for path in possible_files:
            if path.exists():
                corpus_file = path
                break
        
        if not corpus_file:
            raise FileNotFoundError(f"No corpus file found in {self.dataset_dir}")
        
        with open(corpus_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        doc_id = 0
        seen_titles = set()
        
        for item in data:
            contexts = item.get('context', [])
            for context in contexts:
                if isinstance(context, list) and len(context) >= 2:
                    title = context[0]
                    sentences = context[1]
                    
                    if title in seen_titles:
                        continue
                    seen_titles.add(title)
                    
                    if isinstance(sentences, list):
                        text = ' '.join(sentences)
                    else:
                        text = str(sentences)
                    
                    doc = Document(
                        id=str(doc_id),
                        title=title,
                        text=text,
                        metadata={'dataset': 'hotpotqa'}
                    )
                    documents.append(doc)
                    doc_id += 1
        
        logger.info(f"Loaded {len(documents)} documents from HotpotQA corpus")
        return documents
    
    def _load_triviaqa(self, split: str, limit: Optional[int]) -> List[Question]:
        """Load TriviaQA questions."""
        # TriviaQA structure may vary, adapt as needed
        possible_files = [
            self.dataset_dir / f"qa/{split}.json",
            self.dataset_dir / f"{split}.json",
        ]
        
        file_path = None
        for path in possible_files:
            if path.exists():
                file_path = path
                break
        
        if not file_path:
            raise FileNotFoundError(
                f"No TriviaQA file found for split '{split}' in {self.dataset_dir}"
            )
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # TriviaQA format: {"Data": [...]}
        if isinstance(data, dict) and 'Data' in data:
            data = data['Data']
        
        questions = []
        for idx, item in enumerate(data[:limit] if limit else data):
            # TriviaQA has different answer formats
            answer = item.get('Answer', {})
            if isinstance(answer, dict):
                answer_text = answer.get('Value', '') or answer.get('NormalizedValue', '')
            else:
                answer_text = str(answer)
            
            question = Question(
                id=item.get('QuestionId', str(idx)),
                question=item.get('Question', ''),
                answer=answer_text,
                supporting_facts=None,
                type=item.get('QuestionSource'),
                level=None
            )
            questions.append(question)
        
        logger.info(f"Loaded {len(questions)} questions from TriviaQA {split}")
        return questions
    
    def _load_triviaqa_corpus(self) -> List[Document]:
        """Load TriviaQA corpus."""
        # TriviaQA typically has separate evidence files
        evidence_dir = self.dataset_dir / "evidence"
        
        if not evidence_dir.exists():
            # Fallback: extract from questions
            logger.warning("TriviaQA evidence directory not found, extracting from questions")
            return self._extract_triviaqa_corpus_from_questions()
        
        documents = []
        doc_id = 0
        
        # Read evidence files (typically web or wikipedia)
        for evidence_file in evidence_dir.glob("*.json"):
            with open(evidence_file, 'r', encoding='utf-8') as f:
                evidence_data = json.load(f)
            
            for item in evidence_data:
                title = item.get('Title', f'doc_{doc_id}')
                text = item.get('Text', '')
                
                doc = Document(
                    id=str(doc_id),
                    title=title,
                    text=text,
                    metadata={'dataset': 'triviaqa', 'source': evidence_file.stem}
                )
                documents.append(doc)
                doc_id += 1
        
        logger.info(f"Loaded {len(documents)} documents from TriviaQA corpus")
        return documents
    
    def _extract_triviaqa_corpus_from_questions(self) -> List[Document]:
        """Extract corpus from TriviaQA question files."""
        # This is a fallback method
        documents = []
        logger.warning("Using fallback corpus extraction for TriviaQA")
        return documents
    
    def _load_natural_questions(self, split: str, limit: Optional[int]) -> List[Question]:
        """Load Natural Questions."""
        possible_files = [
            self.dataset_dir / f"nq-{split}.json",
            self.dataset_dir / f"{split}.json",
        ]
        
        file_path = None
        for path in possible_files:
            if path.exists():
                file_path = path
                break
        
        if not file_path:
            raise FileNotFoundError(
                f"No Natural Questions file found for split '{split}' in {self.dataset_dir}"
            )
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        questions = []
        for idx, item in enumerate(data[:limit] if limit else data):
            # Natural Questions format varies
            answer = item.get('answer', item.get('short_answers', ['']))
            if isinstance(answer, list):
                answer = answer[0] if answer else ''
            
            question = Question(
                id=item.get('example_id', str(idx)),
                question=item.get('question_text', item.get('question', '')),
                answer=str(answer),
                supporting_facts=None,
                type=None,
                level=None
            )
            questions.append(question)
        
        logger.info(f"Loaded {len(questions)} questions from Natural Questions {split}")
        return questions
    
    def _load_natural_questions_corpus(self) -> List[Document]:
        """Load Natural Questions corpus."""
        # Natural Questions typically includes document text in the question file
        corpus_file = self.dataset_dir / "nq-dev.json"
        
        if not corpus_file.exists():
            corpus_file = self.dataset_dir / "dev.json"
        
        if not corpus_file.exists():
            raise FileNotFoundError(f"No corpus file found in {self.dataset_dir}")
        
        with open(corpus_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        doc_id = 0
        seen_texts = set()
        
        for item in data:
            # Extract document text
            doc_text = item.get('document_text', '')
            if not doc_text or doc_text in seen_texts:
                continue
            
            seen_texts.add(doc_text)
            
            doc = Document(
                id=str(doc_id),
                title=item.get('document_title', f'doc_{doc_id}'),
                text=doc_text,
                metadata={'dataset': 'natural_questions'}
            )
            documents.append(doc)
            doc_id += 1
        
        logger.info(f"Loaded {len(documents)} documents from Natural Questions corpus")
        return documents
