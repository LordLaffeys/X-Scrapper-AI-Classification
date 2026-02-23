import json
import time
from datetime import datetime, timedelta
import random
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager

from dotenv import load_dotenv

load_dotenv()

CT0_TOKEN = os.getenv("CT0_TOKEN", "")
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "")

# --- CONFIGURATION ---
SEARCH_QUERIES = [
    'AI agents tools',
    'AI infrastructure',
    'building an AI',
    'AI Angel Investor',
]

# SCROLL SETTINGS - ADJUST THESE
SCROLL_PIXELS = 2500          # Scroll 2500px at once (big jump)
# SCROLL_DELAY = 1.3            # Wait 1.2s between scrolls (faster)
MAX_SCROLLS_PER_QUERY = 35    # More scrolls allowed
MIN_NEW_PER_QUERY = 70        # Collect more per query
MAX_NEW_PER_QUERY = 70       # Allow more per query
TARGET_TOTAL_ACCOUNTS = 300   # Higher total target
EMPTY_SCROLLS_BEFORE_STOP = 30  # Stop if 4 consecutive empty scrolls

COOKIES = [
    {"name": "auth_token", "value": AUTH_TOKEN, "domain": ".x.com"},
    {"name": "ct0", "value": CT0_TOKEN, "domain": ".x.com"}
]

MAX_FOLLOWERS = 10000
MIN_FOLLOWERS = 50
DAYS_ACTIVE = 30

def parse_twitter_date(date_str):
    try:
        return datetime.strptime(date_str, "%a %b %d %H:%M:%S +0000 %Y")
    except:
        return None

def is_recent_tweet(tweet_date_str, days=DAYS_ACTIVE):
    tweet_date = parse_twitter_date(tweet_date_str)
    if not tweet_date:
        return False
    cutoff = datetime.now() - timedelta(days=days)
    return tweet_date > cutoff

def has_building_signals(user_data, tweet_data=None):
    signals = []
    bio = user_data.get('description', '').lower()
    building_keywords = ['building', 'shipping', 'maker', 'founder', 'creator', 
                        'developer', 'engineer', 'launch', 'product', 'startup',
                        'indie hacker', 'solo founder', 'building in public',
                        'open source', 'github', 'shipping', 'vibe coding']
    
    for keyword in building_keywords:
        if keyword in bio:
            signals.append(f"bio:{keyword}")
    
    pinned = user_data.get('pinned_tweet_ids_str', [])
    if pinned:
        signals.append(f"pinned:{len(pinned)}")
    
    if user_data.get('url'):
        signals.append("has_url")
    
    if tweet_data:
        tweet_text = tweet_data.get('full_text', '').lower()
        for keyword in building_keywords:
            if keyword in tweet_text:
                signals.append(f"tweet:{keyword}")
    
    return signals

def extract_user_from_tweet(tweet_result):
    try:
        core = tweet_result.get('core', {})
        user_results = core.get('user_results', {}).get('result', {})
        
        if not user_results:
            return None, None
            
        legacy = user_results.get('legacy', {})
        core_info = user_results.get('core', {})
        
        # Extract and validate follower count early
        followers = legacy.get('followers_count', 0)
        try:
            followers = int(followers)
        except (ValueError, TypeError):
            followers = 0
        
        # EARLY FILTER: Skip if followers out of range
        if followers > MAX_FOLLOWERS or followers < MIN_FOLLOWERS:
            return None, None  # Return None to signal skip
        
        user_data = {
            'rest_id': user_results.get('rest_id'),
            'handle': core_info.get('screen_name') or legacy.get('screen_name'),
            'name': core_info.get('name') or legacy.get('name'),
            'bio': legacy.get('description', ''),
            'followers': followers,  # Already validated int
            'following': legacy.get('friends_count', 0),
            'statuses_count': legacy.get('statuses_count', 0),
            'created_at': core.get('created_at') or legacy.get('created_at'),
            'location': legacy.get('location', ''),
            'url': legacy.get('url', ''),
            'verified': user_results.get('is_blue_verified', False),
            'pinned_tweets': len(legacy.get('pinned_tweet_ids_str', [])),
            'profile_image': user_results.get('avatar', {}).get('image_url', ''),
        }
        
        return user_data, user_results
    except Exception as e:
        return None, None

def extract_tweet_data(tweet_result):
    try:
        legacy = tweet_result.get('legacy', {})
        note_tweet = tweet_result.get('note_tweet', {}).get('note_tweet_results', {}).get('result', {})
        
        full_text = note_tweet.get('text', legacy.get('full_text', ''))
        
        tweet_data = {
            'tweet_id': legacy.get('id_str'),
            'text': full_text,
            'created_at': legacy.get('created_at'),
            'favorite_count': legacy.get('favorite_count', 0),
            'retweet_count': legacy.get('retweet_count', 0),
            'reply_count': legacy.get('reply_count', 0),
            'quote_count': legacy.get('quote_count', 0),
            'is_recent': is_recent_tweet(legacy.get('created_at', '')),
            'has_media': bool(legacy.get('entities', {}).get('media', [])),
            'has_urls': bool(legacy.get('entities', {}).get('urls', [])),
            'is_quote_status': legacy.get('is_quote_status', False),
        }
        
        return tweet_data
    except:
        return None

def extract_from_user_module(items, query_used):
    users = []
    for item in items:
        try:
            item_content = item.get('item', {}).get('itemContent', {})
            user_results = item_content.get('user_results', {}).get('result', {})
            
            if not user_results:
                continue
                
            legacy = user_results.get('legacy', {})
            core = user_results.get('core', {})
            
            # Extract and validate follower count immediately
            followers = legacy.get('followers_count', 0)
            try:
                followers = int(followers)
            except (ValueError, TypeError):
                followers = 0
            
            # EARLY FILTER: Skip if followers out of range
            if followers > MAX_FOLLOWERS or followers < MIN_FOLLOWERS:
                continue
            
            user_data = {
                'rest_id': user_results.get('rest_id'),
                'handle': core.get('screen_name') or legacy.get('screen_name'),
                'name': core.get('name') or legacy.get('name'),
                'bio': legacy.get('description', ''),
                'followers': followers,  # Already validated int
                'following': legacy.get('friends_count', 0),
                'statuses_count': legacy.get('statuses_count', 0),
                'created_at': legacy.get('created_at'),
                'location': legacy.get('location', ''),
                'url': legacy.get('url', ''),
                'verified': user_results.get('is_blue_verified', False),
                'pinned_tweets': len(legacy.get('pinned_tweet_ids_str', [])),
                'profile_image': user_results.get('avatar', {}).get('image_url', ''),
                'source': 'people_module',
                'found_in_queries': {query_used}
            }
            
            signals = has_building_signals(legacy)
            user_data['building_signals'] = signals
            user_data['signal_count'] = len(signals)
            
            # EARLY FILTER: Skip if no building signals
            if user_data['signal_count'] == 0:
                continue
            
            users.append(user_data)
        except:
            continue
    return users

def process_search_data(data, all_accounts, query_used, query_stats):
    """Process raw search JSON and extract structured data."""
    new_accounts = 0
    updated_accounts = 0
    
    try:
        instructions = data.get('data', {}).get('search_by_raw_query', {}).get('search_timeline', {}).get('timeline', {}).get('instructions', [])
    except:
        return 0, 0
    
    for instruction in instructions:
        if instruction.get('type') != 'TimelineAddEntries':
            continue
            
        entries = instruction.get('entries', [])
        
        for entry in entries:
            content = entry.get('content', {})
            entry_type = content.get('entryType', '')
            
            # Handle People Module
            if entry_type == 'TimelineTimelineModule' and content.get('__typename') == 'TimelineTimelineModule':
                items = content.get('items', [])
                users = extract_from_user_module(items, query_used)
                
                for user in users:
                    handle = user['handle']
                    if handle in all_accounts:
                        all_accounts[handle]['found_in_queries'].add(query_used)
                        query_stats['duplicates'] += 1
                    else:
                        user['tweets'] = []
                        all_accounts[handle] = user
                        new_accounts += 1
                        query_stats['new_this_query'] += 1
            
            # Handle Individual Tweets
            elif entry_type == 'TimelineTimelineItem':
                item_content = content.get('itemContent', {})
                
                if item_content.get('__typename') == 'TimelineTweet':
                    tweet_results = item_content.get('tweet_results', {}).get('result', {})
                    
                    if tweet_results.get('__typename') == 'TweetWithVisibilityResults':
                        tweet_results = tweet_results.get('tweet', {})
                    
                    user_data, raw_user = extract_user_from_tweet(tweet_results)
                    
                    # Skip if user filtered out (None returned)
                    if not user_data:
                        continue
                    
                    tweet_data = extract_tweet_data(tweet_results)
                    
                    if not tweet_data:
                        continue
                    
                    handle = user_data['handle']
                    
                    signals = has_building_signals(raw_user.get('legacy', {}) if raw_user else {}, tweet_data)
                    user_data['building_signals'] = signals
                    user_data['signal_count'] = len(signals)
                    user_data['source'] = 'tweet_search'
                    user_data['found_in_queries'] = {query_used}
                    
                    # EARLY FILTER: Skip if no building signals
                    if user_data['signal_count'] == 0:
                        continue
                    
                    # EARLY FILTER: Check for recent tweet
                    if not tweet_data.get('is_recent', False):
                        continue
                    
                    if handle in all_accounts:
                        all_accounts[handle]['found_in_queries'].add(query_used)
                        existing_tweet_ids = {t['tweet_id'] for t in all_accounts[handle]['tweets']}
                        if tweet_data['tweet_id'] not in existing_tweet_ids:
                            all_accounts[handle]['tweets'].append({
                                **tweet_data,
                                'query': query_used
                            })
                            updated_accounts += 1
                        query_stats['duplicates'] += 1
                    else:
                        user_data['tweets'] = [{
                            **tweet_data,
                            'query': query_used
                        }]
                        all_accounts[handle] = user_data
                        new_accounts += 1
                        query_stats['new_this_query'] += 1
    
    return new_accounts, updated_accounts

def process_logs(driver, all_accounts, query_used, query_stats, processed_request_ids):
    """Process all available logs."""
    logs = driver.get_log("performance")
    new_accounts = 0
    updated_accounts = 0
    
    for entry in logs:
        try:
            message = json.loads(entry["message"])["message"]
            if message["method"] == "Network.responseReceived":
                url = message["params"]["response"]["url"]
                if "SearchTimeline" in url:
                    request_id = message["params"]["requestId"]
                    
                    if request_id in processed_request_ids:
                        continue
                    processed_request_ids.add(request_id)
                    
                    try:
                        body = driver.execute_cdp_cmd("Network.getResponseBody", {"requestId": request_id})
                        data = json.loads(body['body'])
                        n, u = process_search_data(data, all_accounts, query_used, query_stats)
                        new_accounts += n
                        updated_accounts += u
                    except:
                        continue
        except:
            continue
    
    return new_accounts, updated_accounts

def calculate_engagement_score(tweets):
    """Calculate engagement score for a list of tweets."""
    return sum(
        t.get('favorite_count', 0) + t.get('retweet_count', 0) * 2 
        for t in tweets
    )

def save_structured_data(accounts, filename):
    def convert_sets(obj):
        if isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, dict):
            return {k: convert_sets(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_sets(i) for i in obj]
        return obj
    
    # Calculate engagement scores before saving
    for acc in accounts:
        if isinstance(acc, dict):
            acc['engagement_score'] = calculate_engagement_score(acc.get('tweets', []))
            acc['found_in_queries'] = list(acc.get('found_in_queries', set()))
    
    # Sort by signal count and engagement
    accounts.sort(key=lambda x: (x.get('signal_count', 0), x.get('engagement_score', 0)), reverse=True)
    
    cleaned = convert_sets(accounts)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Saved {len(accounts)} accounts to {filename}")

def main():
    options = Options()
    options.set_capability('goog:loggingPrefs', {'performance': 'ALL'})
    options.add_argument("--window-size=1280,800")
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    all_accounts = {}
    processed_request_ids = set()
    
    # Stats tracking
    query_results = []
    
    try:
        driver.get("https://x.com/404")
        for cookie in COOKIES:
            driver.add_cookie(cookie)
        
        for i, query in enumerate(SEARCH_QUERIES, 1):
            print(f"\n{'='*60}")
            print(f"QUERY {i}/{len(SEARCH_QUERIES)}: {query[:50]}...")
            print(f"{'='*60}")
            
            # Per-query stats
            query_stats = {
                'query': query,
                'query_num': i,
                'new_this_query': 0,
                'duplicates': 0,
                'scrolls': 0,
                'filtered_out': 0  # Track how many were filtered
            }
            
            driver.get(f"https://x.com/search?q={query}&f=live")
            time.sleep(5)
            
            # Process initial load
            n, u = process_logs(driver, all_accounts, query, query_stats, processed_request_ids)
            print(f"  Initial: +{n} new | {query_stats['duplicates']} duplicates | Total unique: {len(all_accounts)}")
            
            # Scroll until we get MIN_NEW_PER_QUERY new accounts OR hit MAX_NEW_PER_QUERY
            while query_stats['new_this_query'] < MIN_NEW_PER_QUERY and query_stats['scrolls'] < MAX_SCROLLS_PER_QUERY:
                if len(all_accounts) >= TARGET_TOTAL_ACCOUNTS:
                    break
                
                if query_stats['new_this_query'] >= MAX_NEW_PER_QUERY:
                    break
                
                # BIG FAST SCROLL
                driver.execute_script(f"window.scrollBy(0, {SCROLL_PIXELS});")
                time.sleep(random.uniform(1.0, 2.5))
                query_stats['scrolls'] += 1
                
                n, u = process_logs(driver, all_accounts, query, query_stats, processed_request_ids)
                
                # Track empty scrolls for early exit
                if n == 0:
                    query_stats['empty_scrolls'] = query_stats.get('empty_scrolls', 0) + 1
                    if query_stats['empty_scrolls'] >= EMPTY_SCROLLS_BEFORE_STOP:
                        print(f"  ⛔ {EMPTY_SCROLLS_BEFORE_STOP} empty scrolls, stopping")
                        break
                else:
                    query_stats['empty_scrolls'] = 0
                
                if n > 0:
                    print(f"  Scroll {query_stats['scrolls']}: +{n} new (this query: {query_stats['new_this_query']}, total: {len(all_accounts)})")
            
            # Store stats
            query_results.append({
                'query': query,
                'query_num': i,
                'new_accounts': query_stats['new_this_query'],
                'duplicates_seen': query_stats['duplicates'],
                'scrolls_used': query_stats['scrolls'],
                'total_unique_after': len(all_accounts)
            })
            
            print(f"  Query {i} complete: {query_stats['new_this_query']} new accounts")
        
        # Final summary
        print(f"\n{'='*60}")
        print("COLLECTION SUMMARY")
        print(f"{'='*60}")
        
        for qr in query_results:
            print(f"\nQuery {qr['query_num']}: {qr['new_accounts']} new, {qr['duplicates_seen']} duplicates")
            print(f"  {qr['query'][:60]}...")
        
        print(f"\n{'='*60}")
        print(f"TOTAL UNIQUE ACCOUNTS: {len(all_accounts)}")
        print(f"  (All pre-filtered: {MIN_FOLLOWERS}-{MAX_FOLLOWERS} followers, building signals, recent tweets)")
        print(f"{'='*60}")
        
        # Show breakdown by query source
        print("\n📊 ACCOUNTS BY QUERY SOURCE:")
        query_source_counts = {}
        for handle, acc in all_accounts.items():
            for q in acc.get('found_in_queries', []):
                query_source_counts[q] = query_source_counts.get(q, 0) + 1
        
        for q, count in sorted(query_source_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {count} accounts from: {q[:50]}...")
        
        # Save - no need for separate filter step anymore!
        raw_list = list(all_accounts.values())
        save_structured_data(raw_list, "ai_builders_data.json")
        
        return raw_list
        
    finally:
        driver.quit()

if __name__ == "__main__":
    results = main()
