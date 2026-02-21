# X-Scraper & AI Classifier Pipeline

An automated pipeline to scrape user data from X (Twitter) using Selenium and classify the results using LLMs (Ollama/Gemini).

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) (If using local/cloud Ollama models)
- Chrome Browser (for Selenium)

## Quick Start

### 1. Environment Setup
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configuration
Create a .env file in the root directory:
```
# API Selection
CLASSIFIER_API=ollama # Options: gemini, ollama

# Keys & Tokens
GEMINI_API_KEY=your_key_here
CT0_TOKEN=your_ct0_token_here
AUTH_TOKEN=your_auth_token_here
```

Note: Retrieve CT0 and AUTH tokens from Browser DevTools > Application > Cookies.


### 3. Ollama Setup
If you are using the Ollama classifier:

1. Install Ollama: Download and install from the official website.
2. Local Setup: If you have the hardware specs, download and run your desired model locally.
3. Cloud Setup: * Run ollama in your terminal.
    - Search for "cloud models" on the Ollama website.
    - Connect your browser account to Ollama to use remote models.


### 4. Scraper Configuration
Before running, you can "tinker" with x_scrapper_selenium.py to adjust the data volume and scrolling speed:

| Variable                | Recommended Value | Description                          |
|-------------------------|------------------|--------------------------------------|
| MAX_FOLLOWERS           | 10000            | Upper limit for account size         |
| MIN_FOLLOWERS           | 50               | Lower limit for account size         |
| SCROLL_PIXELS           | 2500             | Jump distance per scroll             |
| MAX_SCROLLS_PER_QUERY   | 35               | Total scrolls allowed per search     |
| TARGET_TOTAL_ACCOUNTS   | 300              | Total accounts to collect            |

### 5. Run Phase
**Step A: Scrape Data**

Run the scraper script in your terminal:
```
python x_scrapper_selenium.py
```

Output: This generates the ai_builders_data.json file.


**Step B: Classify Data**
1. Open classifier_layer.py.
2. Verify Input: Ensure the INPUT filename matches the JSON output from the previous step.
3. Adjust Settings: Set your preferred delay and model. (Note: The code is currently hardcoded set to model="kimi-k2.5:cloud").
4. Run the script:
```
python classifier_layer.py
```
The classifier provides a checkpoint output every 5 records and a final output once the entire process is complete.