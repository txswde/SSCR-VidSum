"""
Counterfactual Generator for Multi-level Causal Interventions
Implements word-level, sentence-level, and semantic-level interventions
"""

import torch
import numpy as np
import random
from typing import List, Tuple, Dict, Optional
import json
import os


class CounterfactualGenerator:
    """
    Generate counterfactual text samples with multi-level interventions
    """
    
    def __init__(self, 
                 strategy: str = 'mixed',
                 intervention_ratio: float = 0.3,
                 cache_dir: Optional[str] = None,
                 use_llm: bool = False,
                 random_seed: int = 42):
        """
        Args:
            strategy: Intervention strategy - 'word', 'sentence', 'semantic', 'mixed'
            intervention_ratio: Ratio of elements to intervene
            cache_dir: Directory to cache pre-generated counterfactuals
            use_llm: Whether to use LLM for semantic-level intervention
            random_seed: Random seed for reproducibility
        """
        self.strategy = strategy
        self.intervention_ratio = intervention_ratio
        self.cache_dir = cache_dir
        self.use_llm = use_llm
        self.random_seed = random_seed
        
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Cache for pre-generated counterfactuals
        self.cache = {}
        if cache_dir and os.path.exists(cache_dir):
            self._load_cache()
    
    def generate_triplet(self, 
                        text_features: torch.Tensor,
                        text_id: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate counterfactual triplet: (original, minimal_intervention, severe_intervention)
        
        Args:
            text_features: Original text features [T, D]
            text_id: Optional text identifier for caching
            
        Returns:
            text_cf_minimal: Minimal intervention counterfactual [T, D]
            text_cf_severe: Severe intervention counterfactual [T, D]
        """
        # Check cache first
        if text_id and text_id in self.cache:
            cached = self.cache[text_id]
            return cached['minimal'], cached['severe']
        
        # Generate based on strategy
        if self.strategy == 'word':
            cf_minimal = self._word_level_intervention(text_features, ratio=0.1)
            cf_severe = self._word_level_intervention(text_features, ratio=0.5)
        elif self.strategy == 'sentence':
            cf_minimal = self._sentence_level_intervention(text_features, ratio=0.2)
            cf_severe = self._sentence_level_intervention(text_features, ratio=0.6)
        elif self.strategy == 'semantic':
            cf_minimal = self._semantic_level_intervention(text_features, intensity='low')
            cf_severe = self._semantic_level_intervention(text_features, intensity='high')
        elif self.strategy == 'mixed':
            # Mix different strategies
            cf_minimal = self._mixed_intervention(text_features, intensity='minimal')
            cf_severe = self._mixed_intervention(text_features, intensity='severe')
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # Cache results
        if text_id:
            self.cache[text_id] = {'minimal': cf_minimal, 'severe': cf_severe}
        
        return cf_minimal, cf_severe
    
    def _word_level_intervention(self, text_features: torch.Tensor, ratio: float) -> torch.Tensor:
        """
        Word-level intervention: perturb individual feature dimensions
        
        Args:
            text_features: [T, D]
            ratio: Ratio of dimensions to perturb
            
        Returns:
            Counterfactual text features [T, D]
        """
        T, D = text_features.shape
        cf_features = text_features.clone()
        
        # Select random dimensions to perturb
        num_perturb = int(D * ratio)
        perturb_dims = np.random.choice(D, num_perturb, replace=False)
        
        # Add Gaussian noise to selected dimensions
        noise = torch.randn(T, num_perturb) * 0.1  # Small perturbation
        cf_features[:, perturb_dims] += noise.to(text_features.device)
        
        return cf_features
    
    def _sentence_level_intervention(self, text_features: torch.Tensor, ratio: float) -> torch.Tensor:
        """
        Sentence-level intervention: shuffle, delete, or duplicate sentences
        
        Args:
            text_features: [T, D]
            ratio: Ratio of sentences to intervene
            
        Returns:
            Counterfactual text features [T, D]
        """
        T, D = text_features.shape
        cf_features = text_features.clone()
        
        num_intervene = max(1, int(T * ratio))
        intervene_indices = np.random.choice(T, num_intervene, replace=False)
        
        for idx in intervene_indices:
            intervention_type = random.choice(['shuffle', 'noise', 'duplicate'])
            
            if intervention_type == 'shuffle' and T > 1:
                # Swap with random sentence
                swap_idx = random.choice([i for i in range(T) if i != idx])
                cf_features[idx], cf_features[swap_idx] = cf_features[swap_idx].clone(), cf_features[idx].clone()
            
            elif intervention_type == 'noise':
                # Add stronger noise
                cf_features[idx] += torch.randn_like(cf_features[idx]) * 0.2
            
            elif intervention_type == 'duplicate' and idx > 0:
                # Duplicate previous sentence
                cf_features[idx] = cf_features[idx - 1].clone()
        
        return cf_features
    
    def _semantic_level_intervention(self, text_features: torch.Tensor, intensity: str) -> torch.Tensor:
        """
        Semantic-level intervention: change high-level semantics
        
        Args:
            text_features: [T, D]
            intensity: 'low' or 'high'
            
        Returns:
            Counterfactual text features [T, D]
        """
        # For now, use linear interpolation with random direction
        # In production, this would use LLM-generated semantic variations
        T, D = text_features.shape
        
        # Generate random semantic direction
        semantic_direction = torch.randn(D).to(text_features.device)
        semantic_direction = semantic_direction / semantic_direction.norm()
        
        # Interpolate based on intensity
        alpha = 0.3 if intensity == 'low' else 0.7
        cf_features = text_features.clone()
        
        # Apply semantic shift
        for t in range(T):
            projection = (cf_features[t] @ semantic_direction) * semantic_direction
            cf_features[t] = (1 - alpha) * cf_features[t] + alpha * projection
        
        return cf_features
    
    def _mixed_intervention(self, text_features: torch.Tensor, intensity: str) -> torch.Tensor:
        """
        Mixed intervention: combine multiple strategies
        
        Args:
            text_features: [T, D]
            intensity: 'minimal' or 'severe'
            
        Returns:
            Counterfactual text features [T, D]
        """
        if intensity == 'minimal':
            # Light word-level + light sentence-level
            cf = self._word_level_intervention(text_features, ratio=0.15)
            cf = self._sentence_level_intervention(cf, ratio=0.1)
        else:  # severe
            # Strong sentence-level + semantic-level
            cf = self._sentence_level_intervention(text_features, ratio=0.5)
            cf = self._semantic_level_intervention(cf, intensity='high')
        
        return cf
    
    def _load_cache(self):
        """Load pre-generated counterfactuals from cache"""
        cache_file = os.path.join(self.cache_dir, 'counterfactual_cache.json')
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                self.cache = json.load(f)
            print(f"Loaded {len(self.cache)} cached counterfactuals")
    
    def save_cache(self):
        """Save generated counterfactuals to cache"""
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            cache_file = os.path.join(self.cache_dir, 'counterfactual_cache.json')
            with open(cache_file, 'w') as f:
                json.dump(self.cache, f)
            print(f"Saved {len(self.cache)} counterfactuals to cache")


class LLMCounterfactualGenerator(CounterfactualGenerator):
    """
    Extended generator using LLM for high-quality semantic interventions
    Requires OpenAI API or local LLM
    """
    
    def __init__(self, *args, llm_model: str = 'gpt-3.5-turbo', **kwargs):
        super().__init__(*args, **kwargs)
        self.llm_model = llm_model
        # TODO: Initialize LLM client
        # self.llm_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    def _semantic_level_intervention(self, text_features: torch.Tensor, intensity: str) -> torch.Tensor:
        """
        Use LLM to generate semantic counterfactuals
        
        This is a placeholder - actual implementation would:
        1. Decode text_features back to text
        2. Call LLM with prompt to generate counterfactual
        3. Re-encode to features
        """
        # Fallback to parent implementation for now
        return super()._semantic_level_intervention(text_features, intensity)


if __name__ == '__main__':
    # Test the generator
    print("Testing CounterfactualGenerator...")
    
    # Create dummy text features
    text_features = torch.randn(10, 512)  # 10 sentences, 512-dim features
    
    # Test different strategies
    for strategy in ['word', 'sentence', 'semantic', 'mixed']:
        print(f"\nTesting strategy: {strategy}")
        generator = CounterfactualGenerator(strategy=strategy)
        
        cf_minimal, cf_severe = generator.generate_triplet(text_features)
        
        # Compute similarity
        sim_minimal = torch.cosine_similarity(text_features, cf_minimal, dim=1).mean()
        sim_severe = torch.cosine_similarity(text_features, cf_severe, dim=1).mean()
        
        print(f"  Minimal intervention similarity: {sim_minimal:.4f}")
        print(f"  Severe intervention similarity: {sim_severe:.4f}")
        print(f"  ✓ Minimal should be > Severe: {sim_minimal > sim_severe}")
