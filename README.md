# Form Guide: Detecting a Cricketer's Form with Hidden Markov Models

ML Mini Project, Semester 5, Unit 4 (Hidden Markov Models).

A batter's form is a hidden state (In form / Out of form). The runs they score are the observations (Low 0–19, Medium 20–49, High 50+). The HMM works out the most likely form at each innings and forecasts the next one.

## Features of the website (`index.html`)
- Innings chart coloured by the Viterbi path, with example batters or your own scores
- 2-state or 3-state model, with editable π, A and B
- State diagram and emission chart
- Step-by-step Viterbi trellis animation
- Forward–backward posterior probabilities and scaling factors
- Baum–Welch (EM) learning with a log-likelihood chart
- Next-innings forecast, theory, syllabus coverage and viva questions

## Run
- Website: open `index.html` in a browser, or host it with GitHub Pages (Settings → Pages → Deploy from branch → main / root).
- Python: `pip install numpy` then `python hmm_cricket_form.py` (or run it in Google Colab).
