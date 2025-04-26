python3 src/scripts/find_arxiv_papers.py \
  --start-date 2025-01-01 \
  --end-date   2025-04-17 \
  --keywords   reasoning  

python3 src/scripts/find_arxiv_papers.py \
  --start-date 2025-01-01 \
  --end-date   2025-04-17 \
  --keywords   GRPO PPO DPO RLHF reasoning

python3 src/scripts/find_arxiv_papers.py \
    --start-date 2025-01-01 \
    --end-date   2025-04-17 \
    --keywords   the \
    --verbose

python3 src/scripts/download_openreview_papers.py --conference neurips --year 2023
python3 src/scripts/download_openreview_papers.py --conference iclr --year 2025

python3 src/scripts/scrape_proceedings.py --conference neurips --year 2024