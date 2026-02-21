# X Data Aquisition & Qualifier Pipeline

## Overview

This project builds a signal-driven pipeline to:
- Collect AI-related accounts from X.com
- Filter for early AI builders
- Classify accounts using a hybrid AI system
- Score them using an “Early AI Signal Strength” metric

---

# Part 1 - X.com Tweets Data Scraper (Selenium + Network Interception)

This scraper collects high-signal data from X.com search results using **Selenium** and Chrome DevTools network logging.

Instead of scraping the DOM or clicking profiles, it intercepts the internal `SearchTimeline` API responses and extracts structured JSON data directly.

The goal is **signal-driven discovery**, not mass crawling.

---

## Why Selenium? (Not Playwright)

In testing, Playwright sessions were detected by X.com and blocked from accessing the home page.

Selenium works more reliably because:

- We inject valid session cookies (`auth_token`, `ct0`)
- We attach to an already authenticated session
- We avoid automating the login flow
- The browser fingerprint behaves closer to a normal user session

This reduces detection risk compared to full login automation.

---

## Why Only Scrape Search Timeline JSON

We intentionally **do not click profiles** or crawl user timelines.

Reasons:

- Profile drilling creates a predictable bot pattern  
  (`open profile → scroll → next profile → repeat`)
- X.com quickly flags sequential timeline harvesting behavior
- Search scrolling resembles organic browsing behavior
- The search endpoint already provides enough signal density

This approach keeps behavior closer to normal user activity.

---

## Anti-Detection Measures

- Randomized scroll delays
- Limited scroll depth per query
- Stop after multiple empty scrolls
- Early filtering to reduce data extraction volume

Without delays, X.com can temporarily block the query engine.  
Randomized timing prevents high-frequency request bursts.

---

## Data Filtering Strategy

Early filters are applied to minimize footprint and improve quality:

- Followers range (min/max threshold)
- Recent tweets only
- Bio + tweet keyword signals (e.g. building, founder, shipping)
- Deduplication across queries

---

## Data Extracted

From SearchTimeline JSON:

- User metadata
- Tweet metadata
- Engagement metrics
- Builder signals
- Query source tracking

Accounts are ranked by:
1. Signal count
2. Engagement score

---

## Design Concept

This is **query-bound, signal extraction**, not aggressive crawling.

The scraper:
- Does not spider networks
- Does not harvest entire timelines
- Does not brute-force profile visits
- Does not automate login flows
It prioritizes:
- Lower behavioral anomaly
- Controlled request volume
- High-quality structured output


# Part 2 - Twitter Account Classifier: AI-Based Classification System

## Overview
Hybrid classification pipeline combining **heuristic rules**, **LLM analysis** (Ollama/Gemini), and **embedding validation** to categorize Twitter accounts and score their "Early AI Signal Strength."

---

## Classification Categories

| Category                | Definition                                                  | Key Signals                                                                                                                                                      |
| ----------------------- | ----------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Early-stage Founder** | Building AI products, shipping MVPs, active development     | "building", "launched", "beta", "startup", "mvp", "product", "shipping", "alpha", low followers (<1k), high self-reference ratio (>0.3), project mentions        |
| **AI Researcher**       | Publishing papers, novel architectures, experiments         | "paper", "arxiv", "research", "experiment", "benchmark", "sota", "novel", "architecture", consistent topic focus                                                 |
| **AI Operator**         | Production ML systems, MLOps, scaling infrastructure        | "production", "deployed", "scaling", "latency", "cost", "optimization", "infrastructure", "mlops", GitHub links, high URL sharing                                |
| **Angel Investor**      | Funding startups, deal flow, portfolio management           | "investor", "angel", "vc", "venture", "portfolio", "funding", "seed", "series a", explicit bio keywords, network size (>3k)                                      |
| **Noise**               | Generic content, aggregators, low signal, insufficient data | "newsletter", "subscribe", "curated", "thread", "breaking", "daily updates", high followers (>50k) + no building signals, sparse data (<2 tweets, <20 bio chars) |

*Disclaimer: These rules are basic implementations for demonstration purposes. They are not fine-tuned for production performance and require precise calibration.

---

## Classification Logic

### 1. Heuristic Filter (Fast Path)
Rules-based pre-filtering for obvious cases:
| Trigger Condition                                                                             | Classification        | Reasons
| --------------------------------------------------------------------------------------------- | --------------------- | ------------------------------------------------- |
| **High followers (>50k)** + no building signals + noise keywords                              | `Noise`               | Large aggregator accounts with generic content    |
| **Investor keywords in bio** + network size (>3k) + investor score >2                         | `Angel Investor`      | Self-declared investors with matching network     |
| **Building query match** + low followers (<1k) + project mentions + self-reference ratio >0.3 | `Early-stage Founder` | Building in public, undiscovered builders         |
| **Researcher score >3** + zero noise + consistency >0.7                                       | `AI Researcher`       | Focused academic content, no promotional spam     |
| **Operator score >3** + GitHub links + URL sharing ratio >0.5                                 | `AI Operator`         | Production engineers sharing code and tools       |
| **Sparse data** (<2 tweets, <20 bio chars)                                                    | `Noise`               | Insufficient signal for meaningful classification |


*depending on the heuristic filter tuning it should be the first filter to classify accounts without LLM

### 2. LLM Classification (Slow Path)
Prompts LLM (Ollama/Gemini) with account bio + sample tweets. Returns:
- `category`: One of 5 categories
- `confidence`: 0-100
- `reasoning`: Human-readable explanation
- `signal_indicators`: Key observations

**For this attempt, I used kimi2.5 ollama cloud model, because it has higher limit and free**

### 3. Embedding Validation
Validates LLM decision using local sentence transformers:
- Encodes account text + reference examples per category
- Cosine similarity score (0-100) checks semantic alignment
- Low embedding score + high LLM confidence = flag for review for illusion checking

---

## Early Signal Score (0-100)

Weighted composite score indicating account quality/promise:

| Component | Weight | Calculation |
|-----------|--------|-------------|
| **Building Velocity** | 25% | Signals count × 20 + self-reference ratio × 30 + GitHub bonus |
| **Technical Specificity** | 20% | Technical terms / (Technical + Noise terms) |
| **Network Position** | 15% | Inverse follower count (&lt;100 = 95, &lt;10k = 50) |
| **Recency** | 10% | % of tweets from last 30 days |
| **Engagement Quality** | 10% | Engagement rate relative to follower count |
| **Classification Confidence** | 20% | Blended LLM + embedding confidence |

**Category Multipliers:**
- Early-stage Founder: **1.25×** (boosted)
- AI Researcher: 1.0×
- AI Operator: 0.9×
- Angel Investor: 0.7×
- Noise: **0.3×** (penalized)

**Bonuses:** +10 for building query match, +15 for low followers + project mentions

**Disclaimer: some of this score calculation idea is combining my idea with AI (chatgpt/claude) ideas and their help, ideally this should be studied and tested further before determining the weights into real production line.**

---
## SUMMARY
Categories: {'early_stage_founder': 161, 'ai_operator': 23, 'angel_investor': 23, 'ai_researcher': 1, 'noise': 71, 'unclear': 1}

APIs used: {'ollama': 261, 'none': 19}, **based on manual human validation and second validation through big models (gpt/gemini) there are alot inaccuracies**

## Prompt Setups (can be tune more)
```
Analyze this Twitter account and classify it into exactly one category.

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
{{"category": "early_stage_founder", "confidence": 75, "reasoning": "...", "signal_indicators": ["..."]}}
```

## Output Schema

```json
"account": {
        "rest_id": "89916725",
        "handle": "hookvm",
        "name": "Pankaj",
        "bio": "Staff Software Engineer | Web3 |Traveller | Politics | Cricket | Credit Card Expert | Life Experiences | Building hookVM | DM for AI agents, Full Stack, MVP dev",
        "followers": 459,
        "following": 421,
        "statuses_count": 1147,
        "verified": true,
        "location": "",
        "url": "https://t.co/VKtqHf0B7q",
        "building_signals": [
          "bio:building",
          "bio:engineer",
          "pinned:1",
          "has_url"
        ],
        "found_in_queries": [
          "building an AI"
        ]
      },
      "classification": {
        "category": "early_stage_founder",
        "early_signal_score": 100,
        "confidence": 52.8,
        "api_used": "ollama",
        "reasoning": "Account shows strong founder signals with explicit 'Building hookVM' in bio, low follower count (459), and technical content discussing AI product architecture decisions. The high Founder score (5) vs minimal Researcher (0) and Operator (1) scores indicates primary identity as builder. Tweet content analyzes engineering trade-offs for AI products (auth systems vs model building), typical of early-stage technical founders architecting MVPs. Bio also solicits DMs for AI agents and MVP development, confirming active building mode.",
        "embedding_validation": 0,
        "signal_indicators": [
          "bio:building hookVM",
          "low_follower_count:459",
          "founder_score:5",
          "tweet_content:building_AI_model",
          "bio:DM_for_AI_agents_MVP_dev",
          "building_signals:bio_engineer_has_url",
          "building_velocity: 80",
          "technical_specificity: 86",
          "network_position: 85",
          "recency: 100",
          "engagement_quality: 100",
          "classification_confidence: 53"
        ],
        "raw_llm_output": {
          "category": "early_stage_founder",
          "confidence": 88,
          "reasoning": "Account shows strong founder signals with explicit 'Building hookVM' in bio, low follower count (459), and technical content discussing AI product architecture decisions. The high Founder score (5) vs minimal Researcher (0) and Operator (1) scores indicates primary identity as builder. Tweet content analyzes engineering trade-offs for AI products (auth systems vs model building), typical of early-stage technical founders architecting MVPs. Bio also solicits DMs for AI agents and MVP development, confirming active building mode.",
          "signal_indicators": [
            "bio:building hookVM",
            "low_follower_count:459",
            "founder_score:5",
            "tweet_content:building_AI_model",
            "bio:DM_for_AI_agents_MVP_dev",
            "building_signals:bio_engineer_has_url",
            "building_velocity: 80",
            "technical_specificity: 86",
            "network_position: 85",
            "recency: 100",
            "engagement_quality: 100",
            "classification_confidence: 53"
          ]
        }
      }
```

## Raw scrapped input data
```json
{
    "rest_id": "89916725",
    "handle": "hookvm",
    "name": "Pankaj",
    "bio": "Staff Software Engineer | Web3 |Traveller | Politics | Cricket | Credit Card Expert | Life Experiences | Building hookVM | DM for AI agents, Full Stack, MVP dev",
    "followers": 459,
    "following": 421,
    "statuses_count": 1147,
    "created_at": null,
    "location": "",
    "url": "https://t.co/VKtqHf0B7q",
    "verified": true,
    "pinned_tweets": 1,
    "profile_image": "https://pbs.twimg.com/profile_images/2005328293407645697/_aIn4iSV_normal.jpg",
    "building_signals": [
      "bio:building",
      "bio:engineer",
      "pinned:1",
      "has_url"
    ],
    "signal_count": 4,
    "source": "tweet_search",
    "found_in_queries": [
      "building an AI"
    ],
    "tweets": [
      {
        "tweet_id": "2024728396841046277",
        "text": "Building an AI model and building a secure login system are two very different problems.\n\nLogin systems aren\u2019t just \u201cemail + password.\u201d\nThey involve:\n\u2022 Identity verification\n\u2022 Fraud detection\n\u2022 Encryption\n\u2022 Password recovery\n\u2022 Compliance\n\u2022 24/7 security monitoring\n\nGoogle/Microsoft logins are used because:\n1\ufe0f\u20e3 They\u2019re secure and battle-tested\n2\ufe0f\u20e3 They reduce friction for users\n3\ufe0f\u20e3 They save time and engineering effort\n\nEven global startups use OAuth instead of building auth from scratch.\n\nUsing Google login doesn\u2019t mean \u201cdependence.\u201d\nIt\u2019s leveraging existing identity infrastructure.\n\nThe real sovereignty question is about:\nData storage, model training, and control not just login buttons.\n\nGood tech isn\u2019t about rebuilding everything.\nIt\u2019s about choosing where to focus.",
        "created_at": "Fri Feb 20 06:10:42 +0000 2026",
        "favorite_count": 29,
        "retweet_count": 0,
        "reply_count": 4,
        "quote_count": 1,
        "is_recent": true,
        "has_media": false,
        "has_urls": false,
        "is_quote_status": false,
        "query": "building an AI"
      }
    ],
    "engagement_score": 29
}
```