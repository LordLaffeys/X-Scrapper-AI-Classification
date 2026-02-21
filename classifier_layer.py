#!/usr/bin/env python3
"""
Twitter Account Classifier - Multi-API Support
Supports: Gemini, Kimi (Moonshot), OpenRouter, Groq, Local Ollama
"""

import json
import os
import sys
import time
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional, Tuple, Any
import requests
from ollama import chat
from dotenv import load_dotenv

load_dotenv()
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False
    print("Warning: sentence-transformers not installed. Embedding validation disabled.")

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False



# ============================================================================
# CONFIGURATION - Set your preferred API here
# ============================================================================

# API Selection: "gemini", "ollama", "none"
PREFERRED_API = os.getenv("CLASSIFIER_API", "ollama")

# API Keys (set via environment variables)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Rate limiting to avoid bottleneck
RATE_LIMITS = {
    "gemini": {"delay": 8, "retries": 3, "retry_delay": 15},     # 60 req/min
    "ollama": {"delay": 8, "retries": 3, "retry_delay": 10},     # cloud/local
    "none": {"delay": 0, "retries": 0, "retry_delay": 0},        # Heuristic only
}

MIN_SIGNAL_SCORE = 60
INPUT_FILE = "ai_builders_data.json"
BATCH_SIZE = 5  # Save every 5 accounts


# ============================================================================
# DATA MODELS
# ============================================================================

class Category(Enum):
    EARLY_STAGE_FOUNDER = "early_stage_founder"
    AI_RESEARCHER = "ai_researcher"
    AI_OPERATOR = "ai_operator"
    ANGEL_INVESTOR = "angel_investor"
    NOISE = "noise"
    UNCLEAR = "unclear"


CATEGORY_ALIASES = {
    "early_stage_founder": Category.EARLY_STAGE_FOUNDER,
    "early_stage_founders": Category.EARLY_STAGE_FOUNDER,
    "founder": Category.EARLY_STAGE_FOUNDER,
    "startup": Category.EARLY_STAGE_FOUNDER,
    "ai_researcher": Category.AI_RESEARCHER,
    "ai_researchers": Category.AI_RESEARCHER,
    "researcher": Category.AI_RESEARCHER,
    "ai_operator": Category.AI_OPERATOR,
    "ai_operators": Category.AI_OPERATOR,
    "operator": Category.AI_OPERATOR,
    "ml_engineer": Category.AI_OPERATOR,
    "engineer": Category.AI_OPERATOR,
    "angel_investor": Category.ANGEL_INVESTOR,
    "angel_investors": Category.ANGEL_INVESTOR,
    "investor": Category.ANGEL_INVESTOR,
    "vc": Category.ANGEL_INVESTOR,
    "noise": Category.NOISE,
}


def parse_category(category_str: str) -> Category:
    if not category_str:
        return Category.UNCLEAR
    normalized = category_str.lower().strip().replace(" ", "_")
    if normalized in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[normalized]
    for alias, cat in CATEGORY_ALIASES.items():
        if alias in normalized or normalized in alias:
            return cat
    return Category.UNCLEAR


@dataclass
class Tweet:
    tweet_id: str
    text: str
    created_at: str
    favorite_count: int
    retweet_count: int
    reply_count: int
    quote_count: int
    is_recent: bool
    has_media: bool
    has_urls: bool
    is_quote_status: bool
    query: str
    
    @property
    def engagement_score(self) -> float:
        return float(self.favorite_count + self.retweet_count * 2 + self.reply_count * 3)


@dataclass
class Account:
    rest_id: str
    handle: str
    name: str
    bio: str
    followers: int
    following: int
    statuses_count: int
    created_at: Optional[str]
    location: str
    url: str
    verified: bool
    pinned_tweets: int
    profile_image: str
    building_signals: List[str]
    signal_count: int
    source: str
    found_in_queries: List[str]
    tweets: List[Tweet]
    scrape_count: int = 1
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None


@dataclass
class ClassificationResult:
    category: Category
    confidence: float
    reasoning: str
    early_signal_score: float
    signal_indicators: List[str]
    heuristic_skipped: bool = False
    embedding_validation: Optional[float] = None
    raw_llm_output: Optional[Dict] = None
    api_used: str = "none"


# ============================================================================
# DATA LOADER
# ============================================================================

class DataLoader:
    def load(self, filepath: str) -> List[Account]:
        print(f"Loading data from {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        if isinstance(raw_data, dict):
            raw_list = list(raw_data.values())
        else:
            raw_list = raw_data
        
        accounts = []
        for raw in raw_list:
            account = self._parse_account(raw)
            if account:
                accounts.append(account)
        
        print(f"Loaded {len(accounts)} accounts from file")
        return accounts
    
    def _parse_account(self, raw: Dict) -> Optional[Account]:
        try:
            rest_id = raw.get('rest_id') or raw.get('id')
            if not rest_id:
                return None
            
            raw_tweets = raw.get('tweets', [])
            tweets = []
            for t in raw_tweets:
                tweets.append(Tweet(
                    tweet_id=t.get('tweet_id', ''),
                    text=t.get('text', t.get('full_text', '')),
                    created_at=t.get('created_at', ''),
                    favorite_count=t.get('favorite_count', 0),
                    retweet_count=t.get('retweet_count', 0),
                    reply_count=t.get('reply_count', 0),
                    quote_count=t.get('quote_count', 0),
                    is_recent=t.get('is_recent', False),
                    has_media=t.get('has_media', False),
                    has_urls=t.get('has_urls', False),
                    is_quote_status=t.get('is_quote_status', False),
                    query=t.get('query', 'unknown')
                ))
            
            tweets.sort(key=lambda x: x.created_at, reverse=True)
            queries = raw.get('found_in_queries', [])
            if isinstance(queries, set):
                queries = list(queries)
            elif not isinstance(queries, list):
                queries = [str(queries)]
            
            timestamps = [t.created_at for t in tweets if t.created_at]
            
            return Account(
                rest_id=str(rest_id),
                handle=raw.get('handle', raw.get('screen_name', '')),
                name=raw.get('name', ''),
                bio=raw.get('bio', raw.get('description', '')),
                followers=raw.get('followers', 0),
                following=raw.get('following', raw.get('friends_count', 0)),
                statuses_count=raw.get('statuses_count', 0),
                created_at=raw.get('created_at'),
                location=raw.get('location', ''),
                url=raw.get('url', ''),
                verified=raw.get('verified', False),
                pinned_tweets=raw.get('pinned_tweets', 0),
                profile_image=raw.get('profile_image', ''),
                building_signals=raw.get('building_signals', []),
                signal_count=raw.get('signal_count', len(raw.get('building_signals', []))),
                source=raw.get('source', 'unknown'),
                found_in_queries=queries,
                tweets=tweets,
                scrape_count=raw.get('scrape_count', 1),
                first_seen=timestamps[-1] if timestamps else None,
                last_seen=timestamps[0] if timestamps else None
            )
        except Exception as e:
            print(f"Error parsing account: {e}")
            return None


# ============================================================================
# FEATURE EXTRACTION (same as before)
# ============================================================================

class FeatureExtractor:
    FOUNDER_INDICATORS = ["building", "launched", "startup", "founder", "mvp", "product", "shipping", "beta", "alpha"]
    RESEARCHER_INDICATORS = ["paper", "arxiv", "research", "experiment", "benchmark", "sota", "novel", "architecture"]
    OPERATOR_INDICATORS = ["production", "deployed", "scaling", "latency", "cost", "optimization", "infrastructure", "mlops"]
    INVESTOR_INDICATORS = ["investor", "angel", "vc", "venture", "portfolio", "funding", "seed", "series a"]
    NOISE_INDICATORS = ["newsletter", "subscribe", "curated", "thread", "breaking", "daily updates"]
    
    GITHUB_PATTERN = re.compile(r'github\.com|t\.co/[a-zA-Z0-9]+')
    PROJECT_PATTERN = re.compile(r'\b[A-Z][a-z]+[A-Z][a-zA-Z]+\b')
    
    def extract_features(self, account: Account) -> Dict[str, Any]:
        tweets_text = " ".join([t.text for t in account.tweets])
        recent_tweets = [t for t in account.tweets if t.is_recent]
        
        return {
            "bio_length": len(account.bio),
            "has_website": bool(account.url),
            "followers": account.followers,
            "following": account.following,
            "follower_following_ratio": account.followers / (account.following + 1),
            "is_new_account": account.created_at is None,
            "query_intents": account.found_in_queries,
            "building_query_match": any("building" in q.lower() for q in account.found_in_queries),
            "building_signals_count": len(account.building_signals),
            "building_signals": account.building_signals,
            "total_tweets": len(account.tweets),
            "recent_tweet_ratio": len(recent_tweets) / max(len(account.tweets), 1),
            "avg_engagement": sum([t.engagement_score for t in account.tweets]) / max(len(account.tweets), 1),
            "has_github_links": bool(self.GITHUB_PATTERN.search(tweets_text)),
            "project_mentions": len(self.PROJECT_PATTERN.findall(tweets_text)),
            "founder_score": self._score_content(tweets_text, self.FOUNDER_INDICATORS),
            "researcher_score": self._score_content(tweets_text, self.RESEARCHER_INDICATORS),
            "operator_score": self._score_content(tweets_text, self.OPERATOR_INDICATORS),
            "investor_score": self._score_content(tweets_text, self.INVESTOR_INDICATORS),
            "noise_score": self._score_content(tweets_text, self.NOISE_INDICATORS),
            "self_reference_ratio": self._self_reference_ratio(account.tweets),
            "url_sharing_ratio": sum(1 for t in account.tweets if t.has_urls) / max(len(account.tweets), 1),
            "consistency_score": self._calculate_consistency(account.tweets),
            "scrape_frequency": account.scrape_count,
            "tweets_sample": self._select_representative_tweets(account.tweets),
            "bio": account.bio,
            "name": account.name,
            "handle": account.handle
        }
    
    def _score_content(self, text: str, indicators: List[str]) -> float:
        text_lower = text.lower()
        return sum(len(re.findall(ind, text_lower)) for ind in indicators)
    
    def _self_reference_ratio(self, tweets: List[Tweet]) -> float:
        if not tweets:
            return 0.0
        indicators = ["i built", "i made", "i created", "my project", "i'm building"]
        count = sum(1 for t in tweets if any(ind in t.text.lower() for ind in indicators))
        return count / len(tweets)
    
    def _calculate_consistency(self, tweets: List[Tweet]) -> float:
        if len(tweets) < 2:
            return 1.0
        queries = [t.query for t in tweets]
        return 1.0 - (len(set(queries)) - 1) / len(queries)
    
    def _select_representative_tweets(self, tweets: List[Tweet], max_n: int = 5) -> List[str]:
        scored = [(sum([t.is_recent * 3, min(t.engagement_score, 5), t.has_urls]), t.text) for t in tweets]
        scored.sort(reverse=True)
        return [t[1] for t in scored[:max_n]]


# ============================================================================
# HEURISTIC CLASSIFIER (same as before)
# ============================================================================

class HeuristicClassifier:
    def classify(self, features: Dict, account: Account) -> Optional[ClassificationResult]:
        
        if features["followers"] > 50000 and features["building_signals_count"] == 0 and features["noise_score"] > 2:
            return ClassificationResult(Category.NOISE, 85.0, "High followers, no building signals", 10.0, 
                                       ["High followers", "Generic content"], True)
        
        if features["investor_score"] > 2 and features["followers"] > 3000 and any(kw in features["bio"].lower() for kw in ["investor", "angel", "vc"]):
            return ClassificationResult(Category.ANGEL_INVESTOR, 80.0, "Explicit investor with network", 45.0,
                                       ["Investor keywords", "Network size"], True)
        
        if features["building_query_match"] and features["followers"] < 1000 and features["project_mentions"] > 0 and features["self_reference_ratio"] > 0.3:
            return ClassificationResult(Category.EARLY_STAGE_FOUNDER, 75.0, "Building in public, low followers", 85.0,
                                       ["Building query", "Low followers", "Self-reference"], True)
        
        if features["researcher_score"] > 3 and features["noise_score"] == 0 and features["consistency_score"] > 0.7:
            return ClassificationResult(Category.AI_RESEARCHER, 70.0, "Research-focused content", 60.0,
                                       ["Research terms", "Consistent"], True)
        
        if features["operator_score"] > 3 and features["has_github_links"] and features["url_sharing_ratio"] > 0.5:
            return ClassificationResult(Category.AI_OPERATOR, 70.0, "Production + code sharing", 55.0,
                                       ["MLOps terms", "GitHub links"], True)
        
        if features["total_tweets"] < 2 and features["bio_length"] < 20:
            return ClassificationResult(Category.NOISE, 60.0, "Insufficient data", 5.0, ["Sparse data"], True)
        
        return None


# ============================================================================
# UNIFIED LLM CLASSIFIER (supports multiple APIs)
# ============================================================================

class LLMClassifier:
    def __init__(self, api_type: str = "ollama"):
        self.api_type = api_type.lower()
        self.config = RATE_LIMITS.get(self.api_type, RATE_LIMITS["none"])
        self.last_request_time = 0
        self.request_count = 0
        self.embedding_model = None
        
        # Initialize embeddings (local, always works)
        if HAS_EMBEDDINGS:
            print("Loading embedding model...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self._init_reference_embeddings()
        
        # Initialize chosen API
        if self.api_type == "gemini" and HAS_GEMINI:
            genai.configure(api_key=GEMINI_API_KEY)
            self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
            print(f" Using Gemini API")
            
        elif self.api_type == "ollama":
            print(f" Using local Ollama)")
            
        else:
            print(f"API '{api_type}' not available, will use heuristics only")
            self.api_type = "none"
    
    def _init_reference_embeddings(self):
        """Initialize reference embeddings for validation."""
        self.reference_texts = {
            Category.EARLY_STAGE_FOUNDER: [
                "Just shipped v0.2 of my AI coding assistant. Looking for beta testers!",
                "Day 45 of building my startup. Here's what I learned about LLM costs.",
                "Launched on Product Hunt today. 100 users in 2 hours!"
            ],
            Category.AI_RESEARCHER: [
                "Our new paper on efficient attention mechanisms is now on arXiv.",
                "Experiment results: sparse MoE scales better than dense at 100B+ params.",
                "Novel architecture for multimodal learning."
            ],
            Category.AI_OPERATOR: [
                "Migrated our inference pipeline to vLLM. Latency dropped 40%.",
                "Cost optimization for GPT-4: how we cut API spend by 60%.",
                "Production incident postmortem: when your RAG pipeline hallucinates."
            ],
            Category.ANGEL_INVESTOR: [
                "Looking at AI infrastructure deals. DM if you're raising pre-seed.",
                "Just wired my first check to an AI agent startup.",
                "Market map: AI coding assistants."
            ],
            Category.NOISE: [
                "AI is changing everything. Here's 10 tools you need to know 🧵",
                "Breaking: New model released.",
                "Subscribe to my newsletter for daily AI updates."
            ]
        }
        
        self.reference_embeddings = {}
        for cat, texts in self.reference_texts.items():
            self.reference_embeddings[cat] = self.embedding_model.encode(texts)
    
    def _wait_for_rate_limit(self):
        """Respect rate limits."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.config["delay"]:
            sleep_time = self.config["delay"] - elapsed
            print(f" Rate limit: waiting {sleep_time:.1f}s...")
            time.sleep(sleep_time)
    
    def classify(self, account: Account, features: Dict) -> ClassificationResult:
        """Route to appropriate API."""
        if self.api_type == "none":
            return ClassificationResult(Category.UNCLEAR, 0.0, "No API configured", 0.0, [], False, None, None, "none")
        
        self._wait_for_rate_limit()
        
        # Build prompt
        prompt = self._build_prompt(account, features)
        
        # Route to API
        for attempt in range(self.config["retries"]):
            try:
                self.last_request_time = time.time()
                self.request_count += 1
                
                if self.api_type == "gemini":
                    raw_output = self._call_gemini(prompt)
                elif self.api_type == "ollama":
                    raw_output = self._call_ollama(prompt)
                else:
                    raise ValueError(f"Unknown API: {self.api_type}")
                
                print(f"    {self.api_type.upper()} request #{self.request_count} successful")
                
                # Parse response
                category = parse_category(raw_output.get("category", "unclear"))
                confidence = raw_output.get("confidence", 50)
                if not isinstance(confidence, (int, float)):
                    confidence = 50
                
                # Validate with embeddings
                embedding_conf = self._validate_embeddings(account, features, category) if self.embedding_model else 50.0
                final_confidence = (confidence * 0.6) + (embedding_conf * 0.4)
                
                return ClassificationResult(
                    category=category,
                    confidence=round(final_confidence, 1),
                    reasoning=raw_output.get("reasoning", "No reasoning"),
                    early_signal_score=0.0,
                    signal_indicators=raw_output.get("signal_indicators", []),
                    heuristic_skipped=False,
                    embedding_validation=round(embedding_conf, 1),
                    raw_llm_output=raw_output,
                    api_used=self.api_type
                )
                
            except Exception as e:
                error_str = str(e)
                print(f"    <> Attempt {attempt + 1} failed: {error_str[:100]}")
                
                # Check if rate limited
                if any(x in error_str.lower() for x in ["429", "rate", "quota", "limit", "exhausted"]):
                    if attempt < self.config["retries"] - 1:
                        wait = self.config["retry_delay"] * (attempt + 1)
                        print(f"    ⏱️  Retrying in {wait}s...")
                        time.sleep(wait)
                        continue
                
                # Non-retryable error
                return ClassificationResult(
                    Category.UNCLEAR, 0.0, f"API error: {error_str[:200]}", 0.0,
                    ["API failed"], False, None, {"error": error_str}, self.api_type
                )
        
        return ClassificationResult(Category.UNCLEAR, 0.0, "Max retries exceeded", 0.0,
                                   ["Max retries"], False, None, None, self.api_type)
    
    def _build_prompt(self, account: Account, features: Dict) -> str:
        """Build classification prompt."""
        tweets_text = "\n".join([f"- {t}" for t in features["tweets_sample"]])
        
        return f"""Analyze this Twitter account and classify it into exactly one category.

ACCOUNT:
- @{account.handle} ({account.name})
- Bio: {account.bio}
- Followers: {account.followers}
- Building signals: {account.building_signals}
- Found via: {account.found_in_queries}

TWEETS:
{tweets_text}

SCORES:
- Founder: {features['founder_score']}, Researcher: {features['researcher_score']}
- Operator: {features['operator_score']}, Investor: {features['investor_score']}
- Self-ref ratio: {features['self_reference_ratio']:.2f}

CATEGORIES:
1. early_stage_founder - Building AI products, sharing progress, low followers
2. ai_researcher - Publishing papers, novel architectures, experiments
3. ai_operator - Production systems, MLOps, scaling, enterprise
4. angel_investor - Funding startups, deal flow, portfolio
5. noise - Generic content, aggregation, low signal

Respond ONLY with JSON:
{{"category": "early_stage_founder", "confidence": 75, "reasoning": "...", "signal_indicators": ["..."]}}"""
    
    def _call_gemini(self, prompt: str) -> Dict:
        """Call Gemini API."""
        response = self.gemini_model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(temperature=0.1, response_mime_type="application/json")
        )
        return json.loads(response.text)
    
    
    import re

    def _call_ollama(self, prompt: str) -> Dict:

        # print(prompt)
        response = chat(
            model="kimi-k2.5:cloud",
            messages=[
                {"role": "system", "content": "You are a strict JSON classifier. Respond ONLY with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            options={
                "temperature": 0.1,
                # "top_p": 0.9,
            }
        )

        content = response.message.content
        
        # Extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
        if json_match:
            content = json_match.group(1)
        else:
            content = content.strip()
        
        # print(content)
        return json.loads(content)
    
    def _validate_embeddings(self, account: Account, features: Dict, category: Category) -> float:
        """Validate classification with local embeddings."""
        if not self.embedding_model or category == Category.UNCLEAR:
            return 50.0
        
        account_text = f"{account.bio} {' '.join(features['tweets_sample'])}"
        account_emb = self.embedding_model.encode([account_text])
        
        ref_embs = self.reference_embeddings.get(category, [])
        if len(ref_embs) == 0:
            return 50.0
        
        similarities = cosine_similarity(account_emb, ref_embs)[0]
        max_sim = float(max(similarities))
        
        # Normalize: 0.3->0, 0.8->100
        return max(0, min((max_sim - 0.3) / 0.5 * 100, 100))


# ============================================================================
# SCORE CALCULATOR (same as before)
# ============================================================================

class ScoreCalculator:
    def calculate(self, account: Account, features: Dict, classification: ClassificationResult) -> ClassificationResult:
        
        components = {
            "building_velocity": min(features["building_signals_count"] * 20 + features["self_reference_ratio"] * 30 + 
                                    features["project_mentions"] * 10 + (20 if features["has_github_links"] else 0), 100),
            "technical_specificity": (features["founder_score"] + features["researcher_score"] + features["operator_score"]) / 
                                    (features["founder_score"] + features["researcher_score"] + features["operator_score"] + 
                                     features["noise_score"] + 1) * 100,
            "network_position": 95 if features["followers"] < 100 else (85 if features["followers"] < 500 else 
                          (70 if features["followers"] < 2000 else (50 if features["followers"] < 10000 else 30))),
            "recency": features["recent_tweet_ratio"] * 100,
            "engagement_quality": 50 + min(features["avg_engagement"] * 10, 50) if features["followers"] < 100 else 
                                min(features["avg_engagement"] / features["followers"] * 10000, 100),
            "classification_confidence": classification.confidence
        }
        
        weights = {"building_velocity": 0.25, "technical_specificity": 0.20, "network_position": 0.15,
                  "recency": 0.10, "engagement_quality": 0.10, "classification_confidence": 0.20}
        
        base_score = sum(components[k] * weights[k] for k in components)
        
        multipliers = {Category.EARLY_STAGE_FOUNDER: 1.25, Category.AI_RESEARCHER: 1.0, 
                      Category.AI_OPERATOR: 0.9, Category.ANGEL_INVESTOR: 0.7, Category.NOISE: 0.3, Category.UNCLEAR: 0.5}
        
        adjusted = base_score * multipliers.get(classification.category, 0.5)
        
        # Bonuses/penalties
        if classification.category == Category.EARLY_STAGE_FOUNDER:
            if features["building_query_match"]: adjusted += 10
            if features["followers"] < 100 and features["project_mentions"] > 0: adjusted += 15
        
        if features["followers"] > 100000 and classification.category != Category.ANGEL_INVESTOR:
            adjusted -= 20
        
        classification.early_signal_score = round(max(0, min(adjusted, 100)), 1)
        classification.signal_indicators.extend([f"{k}: {v:.0f}" for k, v in components.items()])
        
        return classification


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class ClassificationPipeline:
    def __init__(self, api_type: str = PREFERRED_API):
        self.loader = DataLoader()
        self.feature_extractor = FeatureExtractor()
        self.heuristic = HeuristicClassifier()
        self.llm = LLMClassifier(api_type)
        self.scorer = ScoreCalculator()
        self.results = []
    
    def process_file(self, filepath: str) -> List[Tuple[Account, ClassificationResult]]:
        """Process accounts with checkpointing."""
        self.results = []
        
        print(f"\n{'='*60}")
        print("STEP 1: Loading Data")
        print(f"{'='*60}")
        accounts = self.loader.load(filepath)
        
        if not accounts:
            return []
        
        # Estimate time
        est_llm = int(len(accounts) * 0.3)
        config = RATE_LIMITS.get(self.llm.api_type, RATE_LIMITS["none"])
        est_time = (est_llm * config["delay"]) / 60
        
        print(f"\nAPI: {self.llm.api_type.upper()}")
        print(f"Estimated time: {est_time:.1f} min ({est_llm} LLM calls, {config['delay']}s delay)")
        print(f"Heuristic-only: ~{len(accounts) - est_llm} accounts")
        
        for i, account in enumerate(accounts):
            print(f"\n{'='*60}")
            print(f"[{i+1}/{len(accounts)}] @{account.handle}")
            print(f"{'='*60}")
            
            features = self.feature_extractor.extract_features(account)
            classification = self.heuristic.classify(features, account)
            
            if classification is None:
                print("  → Using LLM...")
                classification = self.llm.classify(account, features)
            else:
                print(f"  → Heuristic: {classification.category.value}")
            
            classification = self.scorer.calculate(account, features, classification)
            self.results.append((account, classification))
            
            print(f"\n  ✓ {classification.category.value.upper()} | Score: {classification.early_signal_score:.0f} | API: {classification.api_used}")
            
            if (i + 1) % BATCH_SIZE == 0:
                self._save_checkpoint(i + 1)
        
        self._save_checkpoint(len(accounts), final=True)
        self.results.sort(key=lambda x: x[1].early_signal_score, reverse=True)
        return self.results
    
    def _save_checkpoint(self, count: int, final: bool = False):
        """Save progress (checkpoint = lightweight, final = full)."""
        
        suffix = "final" if final else f"checkpoint_{count}"
        filename = f"classified_{self.llm.api_type}_{suffix}.json"

        if final:
            # Sort before saving final
            self.results.sort(key=lambda x: x[1].early_signal_score, reverse=True)

            output = []
            for account, classification in self.results:
                output.append({
                    "account": {
                        "rest_id": account.rest_id,
                        "handle": account.handle,
                        "name": account.name,
                        "bio": account.bio,
                        "followers": account.followers,
                        "following": account.following,
                        "statuses_count": account.statuses_count,
                        "verified": account.verified,
                        "location": account.location,
                        "url": account.url,
                        "building_signals": account.building_signals,
                        "found_in_queries": account.found_in_queries,
                    },
                    "classification": {
                        "category": classification.category.value,
                        "early_signal_score": classification.early_signal_score,
                        "confidence": classification.confidence,
                        "api_used": classification.api_used,
                        "reasoning": classification.reasoning,
                        "embedding_validation": classification.embedding_validation,
                        "signal_indicators": classification.signal_indicators,
                        "raw_llm_output": classification.raw_llm_output
                    }
                })

        else:
            # Lightweight checkpoint
            output = [{
                "account": {
                    "handle": r[0].handle,
                    "name": r[0].name,
                    "followers": r[0].followers
                },
                "classification": {
                    "category": r[1].category.value,
                    "score": r[1].early_signal_score,
                    "confidence": r[1].confidence,
                    "api": r[1].api_used
                }
            } for r in self.results]

        with open(filename, 'w', encoding="utf-8") as f:
            json.dump({
                "meta": {
                    "api": self.llm.api_type,
                    "processed": count,
                    "time": datetime.now().isoformat(),
                    "total_results": len(self.results)
                },
                "results": output
            }, f, indent=2, ensure_ascii=False)

        print(f" {'Final' if final else 'Checkpoint'} saved: {filename}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Show available APIs
    print(f"\n{'='*60}")
    print("AVAILABLE APIs:")
    print(f"{'='*60}")
    apis = []
    if GEMINI_API_KEY: apis.append("gemini (20 req/min)")
    apis.append("ollama (free)")
    apis.append("none (heuristics only)")
    
    for api in apis:
        marker = " *" if PREFERRED_API in api else ""
        print(f"  - {api}{marker}")
    
    # Check preferred API
    api_key_map = {
        "gemini": GEMINI_API_KEY, 
        "ollama": True, "none": True
    }
    
    if PREFERRED_API not in api_key_map or not api_key_map[PREFERRED_API]:
        print(f"\nPreferred API '{PREFERRED_API}' not configured!")
        print("Switching to 'none' (heuristics only)")
        os.environ["CLASSIFIER_API"] = "none"
    
    # Run
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found!")
        sys.exit(1)
    
    pipeline = ClassificationPipeline()
    results = pipeline.process_file(INPUT_FILE)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    from collections import Counter
    cats = Counter([r[1].category.value for r in results])
    apis_used = Counter([r[1].api_used for r in results])
    
    print("Categories:", dict(cats))
    print("APIs used:", dict(apis_used))
    
    print(f"\nTop 10 by Signal Score:")
    for acc, res in results[:10]:
        print(f"  @{acc.handle}: {res.category.value} ({res.early_signal_score:.0f}) - {res.reasoning[:50]}...")


if __name__ == "__main__":
    main()