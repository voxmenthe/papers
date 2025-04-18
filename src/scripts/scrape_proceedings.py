#!/usr/bin/env python3
"""
scrape_proceedings.py: Scrape paper titles, abstracts, and PDF URLs from NeurIPS static proceedings.

Usage:
  python scrape_proceedings.py --conference neurips --year 2023 [--output-file FILE]
"""
import argparse
import requests
import re
from bs4 import BeautifulSoup
from datetime import datetime
import sys
import openreview
import os
import os.path
from dotenv import load_dotenv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scrape titles, abstracts, and PDF links from conference proceedings"
    )
    parser.add_argument(
        '--conference', choices=['neurips', 'iclr', 'icml'], required=True,
        help="Conference to scrape (neurips, iclr, or icml)"
    )
    parser.add_argument(
        '--year', type=int, default=datetime.now().year,
        help="Year of the conference (defaults to current year)"
    )
    parser.add_argument(
        '--output-file', default=None,
        help="Path to save results (default: {conference}_{year}_papers.txt)"
    )
    return parser.parse_args()


def main():
    # Explicitly load .env from the user's home directory
    dotenv_path = os.path.join(os.path.expanduser("~"), ".env")
    load_dotenv(dotenv_path=dotenv_path, override=True) # Load and override existing env vars

    username = os.getenv("OPENREVIEW_USERNAME")
    password = os.getenv("OPENREVIEW_PASSWORD")
    print(f"DEBUG: Loaded username: {username is not None}, password: {password is not None}") # Check if vars loaded
    
    args = parse_args()
    year = args.year or datetime.now().year
    if args.conference == 'neurips':
        base = 'https://proceedings.neurips.cc' # Updated base URL for NeurIPS
        list_url = f"{base}/paper/{year}" # Updated list URL structure for NeurIPS
    elif args.conference in ['iclr', 'icml']:
        # Placeholder for ICLR/ICML - will need specific logic for these
        base = f"https://{args.conference}.cc" # Example base URL, might need adjustment
        list_url = f"{base}/Conferences/{year}/Schedule?type=Poster" # Example URL structure, needs verification
    else:
        raise ValueError('Unsupported conference: ' + args.conference)

    print(f"Fetching list page: {list_url}")
    resp = requests.get(list_url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')
    entries = [] # Initialize entries list

    if args.conference == 'neurips':
        # NeurIPS structure: proceedings.neurips.cc/paper/YEAR
        # Paper links are usually in list items or divs
        paper_links = soup.find_all('a', href=re.compile(rf'/paper/{year}/file/.*-Paper\.pdf'))
        if not paper_links:
             # Fallback: find links to abstract pages if direct PDF links aren't obvious
             paper_links = soup.find_all('a', href=re.compile(rf'/paper/{year}/hash/.*-Abstract\.html'))
             
        print(f"Found {len(paper_links)} potential paper links on the NeurIPS page.")
        for link in paper_links:
            title = link.text.strip()
            href = link['href']
            
            if href.endswith('-Paper.pdf'):
                pdf_url = base + href
                abstract_url = base + href.replace('-Paper.pdf', '-Abstract.html')
            elif href.endswith('-Abstract.html'):
                abstract_url = base + href
                pdf_url = base + href.replace('-Abstract.html', '-Paper.pdf')
            else: # Skip if the link doesn't match expected patterns
                print(f"Skipping unexpected link format: {href}", file=sys.stderr)
                continue

            # Attempt to fetch abstract from abstract page
            abstract = 'Abstract not found' # Default
            try:
                print(f"  Fetching abstract page: {abstract_url}")
                abs_resp = requests.get(abstract_url)
                if abs_resp.status_code == 200:
                    abs_soup = BeautifulSoup(abs_resp.text, 'html.parser')
                    # Look for abstract text - common patterns include <p> tags or specific divs
                    abstract_tag = abs_soup.find('p', class_='abstract') # Check for specific class first
                    if not abstract_tag:
                         # General heuristic: find the longest <p> tag as it might contain the abstract
                         paragraphs = abs_soup.find_all('p')
                         if paragraphs:
                             abstract_tag = max(paragraphs, key=lambda p: len(p.text.strip()))
                    if abstract_tag:
                         abstract = abstract_tag.text.strip()
                else:
                     print(f"    Warning: Could not fetch abstract page {abstract_url} (status: {abs_resp.status_code})", file=sys.stderr)
            except requests.RequestException as e:
                print(f"    Warning: Error fetching abstract page {abstract_url}: {e}", file=sys.stderr)
            except Exception as e:
                 print(f"    Warning: Error parsing abstract page {abstract_url}: {e}", file=sys.stderr)
            entries.append((title, pdf_url, abstract))
            
    elif args.conference == 'iclr':
        # Step 1: First scrape the ICLR Schedule page to get OpenReview links
        if not username or not password:
            sys.exit("Error: OPENREVIEW_USERNAME and OPENREVIEW_PASSWORD must be set in .env for ICLR.")

        # This approach first gets OpenReview IDs from the Schedule page, then uses the API
        base = 'https://iclr.cc'
        list_url = f"{base}/Conferences/{year}/Schedule?type=Poster"
        print(f"Fetching list page: {list_url}")
        
        # Get the Schedule HTML page
        try:
            resp = requests.get(list_url, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')
            
            # Find all the paper cards (they have a title and OpenReview button)
            paper_elements = soup.find_all('div', class_=lambda x: x and 'maincard' in x and 'poster' in x)
            print(f"Found {len(paper_elements)} potential paper elements on the ICLR page.")
            
            # Initialize OpenReview client for API access
            print(f"Initializing OpenReview client...")
            client = openreview.Client(baseurl='https://api.openreview.net', username=username, password=password)
            
            # Extract OpenReview URLs from the page
            openreview_ids = []
            titles_from_page = []  # Store titles to match with OpenReview data
            
            for i, element in enumerate(paper_elements):
                # Extract the title from the maincard body
                title_div = element.find('div', class_='maincardBody')
                if title_div:
                    title = title_div.text.strip()
                    titles_from_page.append(title)
                    
                    # Find the OpenReview button/link (using title attribute)
                    openreview_btn = element.find('a', title='OpenReview')
                    if openreview_btn and 'href' in openreview_btn.attrs:
                        # Debug: Print the first few elements to verify
                        if i < 5:  # Only show first 5 to avoid flooding output
                            print(f"Debug ({i}): Found OpenReview link: {openreview_btn}")
                        # Extract the forum ID from the OpenReview URL
                        url = openreview_btn['href']
                        # URLs typically like: https://openreview.net/forum?id=PAPER_ID
                        match = re.search(r'id=([^&]+)', url)
                        if match:
                            paper_id = match.group(1)
                            openreview_ids.append(paper_id)
                            print(f"Found paper ID: {paper_id} ({i+1}/{len(paper_elements)})")
                        else:
                            print(f"Could not extract ID from URL: {url}")
                    else:
                        print(f"No OpenReview link found for paper: {title}")
            
            print(f"Extracted {len(openreview_ids)} OpenReview paper IDs")
             
            # Step 2: Fetch detailed paper info using the OpenReview API
            successful_papers = 0
            for i, paper_id in enumerate(openreview_ids):
                try:
                    # Search notes for the forum ID using the 'query' parameter
                    print(f"Searching notes for forum {i+1}/{len(openreview_ids)}: {paper_id}")
                    notes_in_forum = client.search_notes(query={'forum': paper_id})
                    
                    if not notes_in_forum:
                        print(f"  Warning: No notes found for forum {paper_id}")
                        continue

                    # Assume the first note is the main submission
                    note = notes_in_forum[0] 
                    
                    # Extract paper details
                    content = note.content
                    title = content.get('title', 'Title not found').strip()
                    abstract = content.get('abstract', 'Abstract not found').strip()
                    
                    # Extract PDF URL (often in content['pdf'] of the submission note)
                    pdf_relative = content.get('pdf')
                    if pdf_relative:
                        # Ensure it's a valid-looking path/URL before constructing
                        if isinstance(pdf_relative, str) and (pdf_relative.startswith('/') or pdf_relative.startswith('http')):
                            pdf_url = f"https://openreview.net{pdf_relative}" if pdf_relative.startswith('/') else pdf_relative
                        else:
                            print(f"  Warning: Invalid pdf field found for forum {paper_id}: {pdf_relative}")
                            pdf_url = f"https://openreview.net/pdf?id={note.id}" # Fallback using the NOTE id
                    else:
                        # Use the note ID for the fallback PDF link, not the forum ID
                        pdf_url = f"https://openreview.net/pdf?id={note.id}"  
                    
                    entries.append((title, pdf_url, abstract))
                    successful_papers += 1
                    
                except Exception as e:
                    # Use paper_id (forum id) in the error message
                    print(f"Error processing forum {paper_id}: {e}")
             
            print(f"Successfully processed {successful_papers} out of {len(openreview_ids)} papers")
        
        except Exception as e:
            sys.exit(f"Error during OpenReview processing: {e}")

    elif args.conference == 'icml':
        # ICML uses proceedings.mlr.press
        base = 'https://proceedings.mlr.press'
        print(f"Attempting to find ICML volume link for {year} on {base}")
        try:
            toc_resp = requests.get(base, timeout=10)
            toc_resp.raise_for_status()
            toc_soup = BeautifulSoup(toc_resp.text, 'html.parser')
            
            volume_link = None
            # Look for links containing 'International Conference on Machine Learning' and the year
            for a in toc_soup.find_all('a', href=True):
                text = a.text.strip()
                if 'International Conference on Machine Learning' in text and str(year) in text:
                    volume_link = a['href']
                    break
            
            if not volume_link:
                sys.exit(f"Error: Could not find ICML proceedings volume link for {year} on {base}")
            
            # Ensure the link is absolute
            if volume_link.startswith('/'):
                list_url = base + volume_link
            elif volume_link.startswith('http'):
                 list_url = volume_link
            else:
                 list_url = requests.compat.urljoin(base, volume_link)
                 
            print(f"Using ICML volume page: {list_url}")
            resp = requests.get(list_url, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')
            
            # Papers are often within <div class="paper">
            paper_elements = soup.find_all('div', class_='paper')
            print(f"Found {len(paper_elements)} potential paper elements on the ICML page.")
            
            for element in paper_elements:
                title_tag = element.find('p', class_='title')
                if not title_tag:
                    continue
                
                title = title_tag.text.strip()
                link_tag = title_tag.find('a', href=True)
                
                # Defaults
                detail_url_abs = 'Detail URL not found'
                pdf_url = 'PDF link not found'
                abstract = 'Abstract not found'
                
                if link_tag:
                    href = link_tag['href']
                    # Construct absolute detail URL (relative to the *list_url* page)
                    try:
                        detail_url_abs = requests.compat.urljoin(list_url, href)
                    except Exception:
                         print(f"    Warning: Could not construct absolute detail URL from {href} and {list_url}", file=sys.stderr)
                         detail_url_abs = 'Detail URL construction failed'
                     
                # Explicitly find PDF link within the paper element (often in div.links)
                links_div = element.find('div', class_='links')
                pdf_link_tag = None
                if links_div:
                     pdf_link_tag = links_div.find('a', string=re.compile(r'pdf', re.IGNORECASE)) or \
                                      links_div.find('a', href=re.compile(r'\.pdf$', re.IGNORECASE))
                # If not in links_div, check the whole element as fallback
                if not pdf_link_tag:
                     pdf_link_tag = element.find('a', string=re.compile(r'pdf', re.IGNORECASE)) or \
                                      element.find('a', href=re.compile(r'\.pdf$', re.IGNORECASE))

                if pdf_link_tag:
                     pdf_href = pdf_link_tag['href']
                     try:
                         pdf_url = requests.compat.urljoin(list_url, pdf_href) # Relative to list page
                     except Exception:
                         print(f"    Warning: Could not construct absolute PDF URL from {pdf_href} and {list_url}", file=sys.stderr)
                         pdf_url = 'PDF link construction failed'
                elif detail_url_abs.endswith('.html'): # Fallback heuristic only if explicit link fails
                    pdf_url = detail_url_abs[:-5] + '.pdf'
 
                # Fetch detail page for abstract
                if detail_url_abs != 'Detail URL not found' and detail_url_abs != 'Detail URL construction failed' and detail_url_abs.startswith('http'):
                     try:
                         print(f"  Fetching detail page: {detail_url_abs}")
                         detail_resp = requests.get(detail_url_abs, timeout=10)
                         if detail_resp.status_code == 200:
                             detail_soup = BeautifulSoup(detail_resp.text, 'html.parser')
                             # Abstract is usually in <div class="abstract">
                             abstract_tag = detail_soup.find('div', class_='abstract')
                             if abstract_tag:
                                 abstract = abstract_tag.text.strip()
                         else:
                             print(f"    Warning: Could not fetch detail page {detail_url_abs} (status: {detail_resp.status_code})", file=sys.stderr)
                     except requests.RequestException as e:
                         print(f"    Warning: Error fetching detail page {detail_url_abs}: {e}", file=sys.stderr)
                     except Exception as e:
                          print(f"    Warning: Error parsing detail page {detail_url_abs}: {e}", file=sys.stderr)
                 
                if title:
                    entries.append((title, pdf_url, abstract))

        except requests.RequestException as e:
            sys.exit(f"Error fetching ICML page: {e}")
        except Exception as e:
            sys.exit(f"Error processing ICML page: {e}")
            
     
     # Write results
    output_file = args.output_file or f"{args.conference}_{year}_papers.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        for title, url, abst in entries:
            f.write(title + "\n")
            f.write(url + "\n")
            f.write(abst + "\n\n")

    print(f"{len(entries)} papers scraped and saved to {output_file}")


if __name__ == '__main__':
    main()
