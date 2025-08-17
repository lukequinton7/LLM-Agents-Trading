import requests
from bs4 import BeautifulSoup
import csv
from datetime import datetime, timedelta
import time
import re

def generate_date_range(start_date_str, end_date_str, frequency_days=7):
    """
    Generate date strings for Wayback Machine snapshots.
    
    """
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    
    date_list = []
    current_date = start_date
    
    while current_date <= end_date:

        # format for Wayback Machine: YYYYMMDDHHMMSS
        wayback_date = current_date.strftime('%Y%m%d') + "120000"  # Noon
        date_list.append({
            'wayback_date': wayback_date,
            'readable_date': current_date.strftime('%Y-%m-%d')
        })
        current_date += timedelta(days=frequency_days)
    
    return date_list

def scrape_wayback_snapshot(wayback_date, readable_date, session, max_retries=3):
    """
    Scrape articles from a single Wayback Machine snapshot with retry logic.
    """
    wayback_url = f"https://web.archive.org/web/{wayback_date}/https://www.zerohedge.com/"
    
    for attempt in range(max_retries + 1):
        if attempt > 0:
            wait_time = 2 ** attempt  # exp backoff- 2,4,8s
            time.sleep(wait_time)
        
        print(f"\nScraping {readable_date} ({wayback_date})")
        print(f"URL: {wayback_url}")
        
        try:
            response = session.get(wayback_url, timeout=15)
            
            if response.status_code != 200:
                print(f"  Status {response.status_code}")
                if attempt < max_retries:
                    continue
                return []
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # check for actual ZeroHedge content
            title = soup.find('title')
            if not title or 'zerohedge' not in title.get_text().lower():
                print(f"  Not ZeroHedge content")
                if attempt < max_retries:
                    continue
                return []
            
            # if succesful break and process the snapshot
            break
            
        except Exception as e:
            print(f" error : {str(e)}")
            if attempt < max_retries:
                continue
            return []
    
    # extract article links
    article_links = set()
    all_links = soup.find_all('a', href=True)
    
    for link in all_links:
        href = link['href']
        if href and any(pattern in href for pattern in ['/political/', '/markets/', '/economics/', '/geopolitical/', '/energy/', '/technology/', '/news/', '/commodities/']):
            if href.startswith('/web/'):
                # Wayback Machine internal link
                full_wayback_url = f"https://web.archive.org{href}"
                article_links.add(full_wayback_url)
            elif href.startswith('/') and not href.startswith('/web/'):
                # Convert relative URL to Wayback Machine URL
                original_url = f"https://www.zerohedge.com{href}"
                article_wayback_url = f"https://web.archive.org/web/{wayback_date}/{original_url}"
                article_links.add(article_wayback_url)
    
    print(f"  found {len(article_links)} article URLs")
    
    # Extract articles
    articles_extracted = []
    successful_timestamps = 0
    
    for i, article_url in enumerate(list(article_links)):
        try:
            if i % 20 == 0 and i > 0:
                print(f"    Progress: {i}/{len(article_links)} articles processed")
            
            article_response = session.get(article_url, timeout=15)
            if article_response.status_code == 200:
                article_soup = BeautifulSoup(article_response.content, 'html.parser')
                
                headline = extract_headline(article_soup)
                article_body = extract_article_body(article_soup)
                article_datetime = extract_datetime_from_html(article_soup, article_url)
                
                # Use extracted datetime or fallback to snapshot date with time
                if article_datetime:
                    datetime_str = article_datetime.strftime('%Y-%m-%d %H:%M:%S')
                    successful_timestamps += 1
                else:
                    # Fallback: use snapshot date at noon
                    snapshot_datetime = datetime.strptime(wayback_date[:8], '%Y%m%d').replace(hour=12, minute=0, second=0)
                    datetime_str = snapshot_datetime.strftime('%Y-%m-%d %H:%M:%S')
                
                articles_extracted.append({
                    'datetime': datetime_str,
                    'headline': headline,
                    'url': article_url,
                    'body': article_body
                })
            
            time.sleep(0.2)  # avoid overwhelming the server
            
        except Exception as e:
            continue
    
    print(f"Extracted {len(articles_extracted)} articles ({successful_timestamps} with real timestamps)") #print no. of articles with time stamps
    return articles_extracted

def extract_datetime_from_html(soup, article_url):
    """
    Extract datetime by searching for common patterns in the HTML.
    This finds the timestamps that exist in Wayback Machine snapshots.
    """
    # raw HTML content
    article_html = str(soup)
    
    # Search for the datetime pattern we found in investigation
    patterns = [
        r'202[0-9]-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}',  # ISO datetime
        r'202[0-9]-[0-9]{2}-[0-9]{2}\s[0-9]{2}:[0-9]{2}:[0-9]{2}', # space-separated 
        r'"datePublished":\s*"([^"]+)"',                            # JSON datePublished  
        r'"published":\s*"([^"]+)"',                                # JSON published
        r'data-time[=\s]*["\']([^"\']+)["\']',                     # data-time attributes
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, article_html, re.IGNORECASE)
        if matches:
            # Take the first match
            datetime_str = matches[0]
            
            try:
                # handle different formats 
                if 'T' in datetime_str:
                    # ISO format: 2024-06-01T05:10:00
                    if len(datetime_str) == 19:  # YYYY-MM-DDTHH:MM:SS
                        return datetime.strptime(datetime_str, '%Y-%m-%dT%H:%M:%S')
                    elif '+' in datetime_str or 'Z' in datetime_str:
                        # Handle timezone
                        clean_datetime = datetime_str.replace('Z', '+00:00')
                        article_datetime = datetime.fromisoformat(clean_datetime)
                        return article_datetime.replace(tzinfo=None)
                elif ' ' in datetime_str:
                    # Space format: 2024-06-01 05:10:00
                    return datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
                else:
                    # date only, add noon
                    date_part = datetime.strptime(datetime_str[:10], '%Y-%m-%d')
                    return date_part.replace(hour=12, minute=0, second=0)
                    
            except Exception as e:
                continue
    
    return None

def scrape_zerohedge_full_range(start_date, end_date, frequency_days=1):
    """
    Scrape ZeroHedge articles for full date range using Wayback Machine.... frequency_days: Sampling frequency (7=weekly, 3=every 3 days, etc.)

    """

    print(f"Date range: {start_date} to {end_date}")
    print(f"Sampling frequency- Every {frequency_days} days")

    
    # Generate date list
    date_list = generate_date_range(start_date, end_date, frequency_days)
    print(f"Will process {len(date_list)} snapshots")
    
    # Setup session
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })
    
    # CSV setup
    csv_filename = f"zerohedge_full_range_{start_date}_to_{end_date}.csv"
    csv_headers = ['publication_datetime', 'article_headline', 'article_url', 'article_body']
    
    all_articles = []
    successful_snapshots = 0
    total_real_timestamps = 0
    
    # Process each snapshot
    for i, date_info in enumerate(date_list):
        print(f"\n--- Snapshot {i+1}/{len(date_list)} ---")
        
        articles = scrape_wayback_snapshot(
            date_info['wayback_date'], 
            date_info['readable_date'], 
            session
        )
        
        if articles:
            all_articles.extend(articles)
            successful_snapshots += 1
            # Count articles with real timestamps (not 12:00:00)
            real_timestamps = sum(1 for a in articles if not a['datetime'].endswith('12:00:00'))
            total_real_timestamps += real_timestamps
        
        # Progress update
        if i % 5 == 0 and i > 0:
            print(f"\n🔄 PROGRESS UPDATE:")
            print(f"   Snapshots processed: {i+1}/{len(date_list)}")
            print(f"   Successful snapshots: {successful_snapshots}")
            print(f"   Total articles collected: {len(all_articles)}")
            print(f"   Articles with real timestamps: {total_real_timestamps}")
        
        # delay between snapshots
        time.sleep(0.2)
    
    # Save to CSV
    print(f"\n" + "=" * 70)
    print("SAVING RESULTS...")
    
    if all_articles:
        # Remove duplicates based on headline
        seen_headlines = set()
        unique_articles = []
        
        for article in all_articles:
            headline_key = article['headline'].lower().strip()
            if headline_key not in seen_headlines and len(headline_key) > 10:
                seen_headlines.add(headline_key)
                unique_articles.append(article)
        
        # Sort by datetime
        unique_articles.sort(key=lambda x: x['datetime'])
        
        with open(csv_filename, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(csv_headers)
            
            for article in unique_articles:
                writer.writerow([
                    article['datetime'],
                    article['headline'],
                    article['url'],
                    article['body']
                ])
        
        # timestamp success rate
        
        print(f"   Total articles extracted: {len(all_articles)}")

        print(f"   Saved to: {csv_filename}")
        
        # Date coverage summary
        dates_covered = sorted(set(article['datetime'][:10] for article in unique_articles))  # just date part
        print(f"   Date coverage: {dates_covered[0]} to {dates_covered[-1]}")
        print(f"   Months covered: {len(set(d[:7] for d in dates_covered))}")
        
    else:
        print(f"No articles extracted")
    
    print("=" * 70)

def extract_headline(soup):
    """Extract headline from article."""
    selectors = ['h1', 'h1.title', 'h1.entry-title']
    for selector in selectors:
        element = soup.select_one(selector)
        if element:
            headline = element.get_text(strip=True)
            if headline and len(headline) > 5:
                return headline
    return "No headline found"

def extract_article_body(soup):
    """Extract main article content."""
    selectors = ['div.prose', 'div[class*="content"]', 'article', 'main', '.entry-content']
    
    for selector in selectors:
        content_div = soup.select_one(selector)
        if content_div:
            paragraphs = content_div.find_all('p')
            meaningful_paragraphs = [
                p.get_text(strip=True) 
                for p in paragraphs 
                if p.get_text(strip=True) and len(p.get_text(strip=True)) > 20
            ]
            if meaningful_paragraphs:
                return "\n\n".join(meaningful_paragraphs[:50])  # Limit for file size
    
    # Fallback
    paragraphs = soup.find_all('p')
    long_paragraphs = [
        p.get_text(strip=True) 
        for p in paragraphs 
        if len(p.get_text(strip=True)) > 50
    ]
    return "\n\n".join(long_paragraphs[:50]) if long_paragraphs else "Could not extract content"


if __name__ == "__main__":

    # configuration
    START_DATE = "2024-03-31"
    END_DATE = "2024-04-30"
    FREQUENCY_DAYS = 1  # daily sampling

    
    scrape_zerohedge_full_range(START_DATE, END_DATE, FREQUENCY_DAYS)