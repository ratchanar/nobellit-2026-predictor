# NobelLit 2026 Predictor — Version 2

An educational Streamlit dashboard combining:

- official historical Nobel Prize Literature data
- public 2026 market/odds signals
- historical similarity
- literary profile scoring
- probability and odds calculations
- interactive sensitivity analysis
- explainable prediction

## Important

This is **not an official Nobel prediction**. The Swedish Academy's shortlist and deliberations are confidential. Public odds are speculative.

Version 2 deliberately avoids pretending that a normal supervised logistic-regression model can be trained from the available data.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m streamlit run app.py
```

## Data sources

Historical Nobel data:
https://www.nobelprize.org/organization/developer-zone-2/

Official Nobel API:
https://api.nobelprize.org/2.1/nobelPrizes

2026 public market/odds context:
The Guardian, 7 October 2026:
https://www.theguardian.com/books/2026/oct/07/anne-carson-and-can-xue-favourites-to-win-nobel-prize-in-literature

Kalshi Nobel Literature 2026 market:
https://kalshi.com/markets/kxnobellit/nobel-literature-prize/kxnobellit-26

## Project structure

- `app.py`
- `candidates_2026.csv`
- `requirements.txt`
- `.gitignore`
- `README.md`
