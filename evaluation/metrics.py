"""
Evaluation Metrics for Video Captioning
Implements BLEU, METEOR, CIDEr, ROUGE-L
"""

import numpy as np
from typing import List, Dict
from collections import Counter, defaultdict
import math


class CaptionMetrics:
    """Evaluation metrics for video-to-text generation"""
    
    @staticmethod
    def bleu_score(reference: str, candidate: str, n: int = 4) -> float:
        """Calculate BLEU-N score"""
        
        ref_tokens = reference.lower().split()
        cand_tokens = candidate.lower().split()
        
        if len(cand_tokens) == 0:
            return 0.0
        
        # Calculate n-gram precisions
        precisions = []
        for i in range(1, n + 1):
            ref_ngrams = Counter(tuple(ref_tokens[j:j+i]) for j in range(len(ref_tokens)-i+1))
            cand_ngrams = Counter(tuple(cand_tokens[j:j+i]) for j in range(len(cand_tokens)-i+1))
            
            overlap = sum((cand_ngrams & ref_ngrams).values())
            total = sum(cand_ngrams.values())
            
            if total == 0:
                precisions.append(0)
            else:
                precisions.append(overlap / total)
        
        # Geometric mean
        if min(precisions) == 0:
            return 0.0
        
        geo_mean = math.exp(sum(math.log(p) for p in precisions) / n)
        
        # Brevity penalty
        bp = 1.0 if len(cand_tokens) > len(ref_tokens) else math.exp(1 - len(ref_tokens)/len(cand_tokens))
        
        return bp * geo_mean
    
    @staticmethod
    def meteor_score(reference: str, candidate: str) -> float:
        """Simplified METEOR score (unigram F-score)"""
        
        ref_tokens = set(reference.lower().split())
        cand_tokens = set(candidate.lower().split())
        
        if len(cand_tokens) == 0:
            return 0.0
        
        overlap = len(ref_tokens & cand_tokens)
        
        precision = overlap / len(cand_tokens) if len(cand_tokens) > 0 else 0
        recall = overlap / len(ref_tokens) if len(ref_tokens) > 0 else 0
        
        if precision + recall == 0:
            return 0.0
        
        f_score = (10 * precision * recall) / (9 * precision + recall)
        
        return f_score
    
    @staticmethod
    def cider_score(references: List[str], candidate: str) -> float:
        """CIDEr score (consensus-based evaluation)"""
        
        def compute_tf_idf(sentences):
            """Compute TF-IDF for all documents"""
            ngrams_all = defaultdict(int)
            ngrams_per_doc = []
            
            for sent in sentences:
                tokens = sent.lower().split()
                ngrams = Counter(tuple(tokens[i:i+4]) for i in range(len(tokens)-3))
                ngrams_per_doc.append(ngrams)
                for ng in ngrams:
                    ngrams_all[ng] += 1
            
            # IDF
            n_docs = len(sentences)
            idf = {ng: math.log(n_docs / count) for ng, count in ngrams_all.items()}
            
            # TF-IDF
            tf_idf_docs = []
            for ngrams in ngrams_per_doc:
                tf_idf = {ng: count * idf.get(ng, 0) for ng, count in ngrams.items()}
                tf_idf_docs.append(tf_idf)
            
            return tf_idf_docs
        
        # Compute TF-IDF
        all_sentences = references + [candidate]
        tf_idf = compute_tf_idf(all_sentences)
        
        cand_tfidf = tf_idf[-1]
        ref_tfidfs = tf_idf[:-1]
        
        # Cosine similarity
        scores = []
        for ref_tfidf in ref_tfidfs:
            dot_product = sum(cand_tfidf.get(ng, 0) * ref_tfidf.get(ng, 0) for ng in set(cand_tfidf) | set(ref_tfidf))
            norm_cand = math.sqrt(sum(v**2 for v in cand_tfidf.values()))
            norm_ref = math.sqrt(sum(v**2 for v in ref_tfidf.values()))
            
            if norm_cand * norm_ref > 0:
                scores.append(dot_product / (norm_cand * norm_ref))
            else:
                scores.append(0.0)
        
        return np.mean(scores) if scores else 0.0
    
    @staticmethod
    def rouge_l(reference: str, candidate: str) -> float:
        """ROUGE-L (Longest Common Subsequence)"""
        
        ref_tokens = reference.lower().split()
        cand_tokens = candidate.lower().split()
        
        m, n = len(ref_tokens), len(cand_tokens)
        
        # DP for LCS
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_tokens[i-1] == cand_tokens[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        lcs_length = dp[m][n]
        
        if m == 0 or n == 0:
            return 0.0
        
        precision = lcs_length / n
        recall = lcs_length / m
        
        if precision + recall == 0:
            return 0.0
        
        f_score = (2 * precision * recall) / (precision + recall)
        
        return f_score
    
    @classmethod
    def evaluate(cls, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Evaluate all metrics"""
        
        assert len(predictions) == len(references), "Length mismatch"
        
        bleu_scores = []
        meteor_scores = []
        rouge_scores = []
        cider_scores = []
        
        for pred, ref in zip(predictions, references):
            bleu_scores.append(cls.bleu_score(ref, pred))
            meteor_scores.append(cls.meteor_score(ref, pred))
            rouge_scores.append(cls.rouge_l(ref, pred))
            cider_scores.append(cls.cider_score([ref], pred))
        
        return {
            'BLEU-4': np.mean(bleu_scores),
            'METEOR': np.mean(meteor_scores),
            'ROUGE-L': np.mean(rouge_scores),
            'CIDEr': np.mean(cider_scores),
            'num_samples': len(predictions)
        }


if __name__ == "__main__":
    # Test metrics
    references = [
        "Flood waters rising rapidly in residential area",
        "Wildfire spreading across forest terrain",
        "Earthquake damage to urban infrastructure"
    ]
    
    predictions = [
        "Rising flood threatening homes and buildings",
        "Forest fire advancing through vegetation",
        "Structural damage from seismic activity"
    ]
    
    metrics = CaptionMetrics.evaluate(predictions, references)
    
    print("📊 Evaluation Results:")
    for metric, score in metrics.items():
        if metric != 'num_samples':
            print(f"  {metric}: {score:.4f}")
