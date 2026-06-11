# World Cup 2026 Monte Carlo Predictor

A from-scratch probabilistic model for the 2026 FIFA World Cup. It rates all 48 teams from 150 years of international results, simulates the real tournament bracket tens of thousands of times, and reports each team's probability of lifting the trophy. The rating engine is validated out-of-sample: it predicts held-out international matches at 60.6% accuracy, 15% better on log-loss than a base-rate baseline. Predictions were locked on June 10, 2026, before the opening match.

**[Live dashboard](#)** (GitHub Pages): simulate one tournament and watch the bracket resolve, or run 10,000 simulations for the full probability board.

## Headline result (locked pre-kickoff)

| Team | Win probability |
|------|----------------|
| Spain | 19.6% |
| Argentina | 12.7% |
| France | 10.7% |
| England | 7.8% |
| Brazil | 4.8% |
| Portugal | 4.4% |
| Colombia | 4.2% |
| Netherlands | 4.2% |
| Germany | 3.9% |
| Ecuador | 3.4% |

Spain enters as a clear but not dominant favorite. The 19.6% reflects a simple truth the simulation makes concrete: even the strongest team must win five straight knockout matches, and it loses roughly a third of each, so the most likely single outcome is still that the favorite does not win.

## How it works

The model has three layers, each addressing a specific weakness of the one before.

**1. Elo ratings.** Every international match since 1872 (about 49,000 games) updates a running strength rating per team, with the update size scaled by match importance (World Cup highest, then continental championships, then qualifiers and Nations League, then friendlies) and by goal margin. Elo is used because it is, by construction, an opponent-adjusted record of recent form: a team's rating already encodes who it beat and how recently.

**2. Confederation calibration.** Raw Elo systematically misrates teams across confederations, because a team can accumulate rating by beating weak regional opponents and rarely get tested outside its confederation. Measured on inter-confederation matches since 2014, UEFA, CONMEBOL, and CAF teams beat their Elo expectation when they leave home, while AFC, CONCACAF, and OFC teams fall short. The model fits one Elo offset per confederation so those inter-confederation residuals go to zero. This correction is derived entirely from the match data, with no external inputs.

**3. Monte Carlo tournament simulation.** Calibrated ratings drive an independent-Poisson goals model (rating gap to expected goals, then sampled scorelines). The simulation uses the actual FIFA bracket: the 16 official Round-of-32 slot definitions, the best-eight-third-place qualification rule, and the real feed-forward tree through to the final. Running it tens of thousands of times turns per-match probabilities into tournament outcomes.

## Validation

The model is tested out-of-sample with a prequential (walk-forward) backtest: marching through every international match in date order, it predicts each held-out match before seeing the result, then scores it. No result is ever used to predict itself.

Over 3,594 held-out matches since January 2023:

| | Log-loss | Accuracy |
|---|---|---|
| Model | 0.894 | 60.6% |
| Base-rate baseline | 1.054 | 47.2% |

The model is 15% better on log-loss and 13 points more accurate than a no-skill predictor, on a hard three-way problem (win/draw/loss) it never trained on. The confederation calibration was fit on pre-2023 data only and tested on the holdout: on the inter-confederation matches it actually affects (the kind a World Cup is made of), it improves log-loss by 3.5%, confirming it is a genuine correction rather than overfitting.

Reproduce with `python wc2026_backtest.py`.

## Run it

```bash
pip install pandas numpy
curl -sL "https://raw.githubusercontent.com/martj42/international_results/master/results.csv" -o results.csv
python wc2026_bracket.py     # build ratings, simulate, print the win-probability board
python wc2026_backtest.py    # reproduce the out-of-sample validation
```

The dashboard (`index.html`) is a single self-contained file with the ratings baked in. Open it locally or host it on GitHub Pages, no backend required.

## Honest limitations

This is a v1. The ratings are validated (see above), but the model has known simplifications:

- **The goals model is independent Poisson.** It does not yet use a Dixon-Coles correction for the correlation between low scorelines, and the rating-to-goals sensitivity parameter is set by hand, not fit.
- **Validation is at the match level, not the tournament level.** The match backtest is strong, but the model has not yet been retrodicted against full past tournaments (predict the 2022 World Cup bracket, etc.).
- **Third-place slot assignment** respects each Round-of-32 slot's allowed-group pool via constrained matching, rather than reproducing FIFA's exact 495-row Annex C lookup. The effect on aggregate probabilities is negligible, but it is a simplification.
- **No squad, injury, or in-tournament information.** Ratings are frozen at the pre-kickoff snapshot.

## Roadmap

- Tournament-level retrodiction (predict the 2022 World Cup, Euro 2024, Copa América 2024 and score the bracket)
- Dixon-Coles goals model as a second engine, ensembled with the current one
- Fit the rating-to-goals sensitivity rather than setting it by hand
- Live Bayesian updating of ratings as group-stage results come in

## Data

International results from [martj42/international_results](https://github.com/martj42/international_results), 1872 to present.
