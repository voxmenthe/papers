#!/usr/bin/env python3
"""
download_openreview_papers.py: Download OpenReview PDFs for NeurIPS or ICLR submissions.

Usage:
  python download_openreview_papers.py --conference neurips [--year YEAR] [--output-dir DIR]
"""
import argparse
from datetime import datetime
from pathlib import Path
import openreview
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path.home() / '.env')
import os
import sys
import requests


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download OpenReview submission PDFs for NeurIPS or ICLR"
    )
    parser.add_argument(
        '--conference', choices=['neurips', 'iclr'], required=True,
        help="Which conference to download (neurips or iclr)"
    )
    parser.add_argument(
        '--year', type=int, default=None,
        help="Conference year (defaults to current year)"
    )
    parser.add_argument(
        '--output-dir', default=None,
        help="Directory to save PDFs (default= openreview_{conf}_{year})"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    year = args.year or datetime.now().year
    conf = args.conference.lower()
    group_prefix = "NeurIPS.cc" if conf == 'neurips' else "ICLR.cc"
    conf_group = f"{group_prefix}/{year}/Conference"
    invitation = f"{conf_group}/-/Camera_Ready"
    print(f"Fetching camera-ready submissions for {conf_group}")

    # authenticate via environment variables
    username = os.environ.get("OPENREVIEW_USERNAME")
    password = os.environ.get("OPENREVIEW_PASSWORD")
    if not (username and password):
        sys.exit("Error: OPENREVIEW_USERNAME and OPENREVIEW_PASSWORD must be set in the environment")
    client = openreview.Client(baseurl="https://api.openreview.net", username=username, password=password)

    notes = client.get_notes(invitation=invitation)
    total = len(notes)
    if total == 0:
        print("No submissions found. Check conference and year.")
        return

    outdir = args.output_dir or f"openreview_{conf}_{year}"
    Path(outdir).mkdir(parents=True, exist_ok=True)

    for note in notes:
        pdf_url = note.content.get('pdf')
        if not pdf_url:
            print(f"Skipping {note.id}: no PDF link")
            continue
        paper_id = note.id.replace('/', '_')
        dest = Path(outdir) / f"{paper_id}.pdf"
        print(f"Downloading {paper_id} -> {dest.name}")
        resp = requests.get(pdf_url)
        resp.raise_for_status()
        with open(dest, 'wb') as f:
            f.write(resp.content)

    print(f"Downloaded {total} papers to {outdir}")


if __name__ == '__main__':
    main()
