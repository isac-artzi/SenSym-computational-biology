# Setting up the celldetect Python laboratory

Everything below happens inside the `python/` directory of the repo.
Total time: about ten minutes, once.

## 1. Check your Python

```
python3 --version
```

You want **3.9 or newer**. If `python3` is not found: install Python from
<https://www.python.org/downloads/> (macOS/Windows) or your package manager
(Linux: `sudo apt install python3 python3-pip python3-venv`). On Windows the
command may be `py` instead of `python3` — substitute accordingly throughout.

## 2. Install the four dependencies

Option A — a virtual environment (recommended: keeps the course's packages
out of your system Python):

```
cd python
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Re-run the `activate` line in every new terminal before working. Your prompt
shows `(.venv)` when it is active.

Option B — plain pip, no venv (fine on a machine that is yours alone):

```
cd python
pip install -r requirements.txt      # or: pip3 install -r requirements.txt
```

What you just installed, and why each one is there:

| Package | Why |
|---|---|
| numpy | arrays, random generators — everything |
| scipy | `erf` / `erfinv` (the Gaussian CDF and its inverse) and `gammaln` (log-factorials without overflow) |
| matplotlib | the weekly figures, saved as PNG |
| pytest | the test runner — the tests define "done" |

`requirements-optional.txt` holds scanpy (needed **once**, in Week 10, to
download PBMC 3k) and scikit-learn (an optional cross-check in Week 8).
Neither is needed to start, and the project's numbers must come from your own
code, not from scikit-learn.

## 3. Smoke test

```
python3 -c "import numpy, scipy, matplotlib; print('imports ok')"
python3 -c "import celldetect.counting; print('celldetect ok')"
```

(Run from inside `python/` — the package is imported from the current
directory; there is nothing to "install" for celldetect itself.)

## 4. Run the tests

```
pytest tests/test_counting.py
```

Expect a wall of failures with `NotImplementedError` — **that is correct and
intended**. Each scaffold function raises until you implement it; the tests
turning green one by one IS the course workflow. Run the whole suite with
`pytest tests/`.

A useful habit: `pytest tests/ -q --tb=no` gives you a one-line-per-test
scoreboard of how much of the semester is done.

## 5. Run your first experiment

After Week 1's two functions are implemented:

```
python experiments/week01_counting_noise.py
```

It prints a short summary and saves PNGs into `python/figures/` — open them.
Every week has a driver script like this; they never open windows, they only
save files.

## Colab fallback (no local Python? Chromebook? locked-down laptop?)

Google Colab runs everything in the browser:

1. Upload the whole `python/` folder to your Google Drive.
2. In a new notebook at <https://colab.research.google.com>, mount Drive and
   enter the folder:

   ```
   from google.colab import drive
   drive.mount('/content/drive')
   %cd /content/drive/MyDrive/python
   ```

3. Install the dependencies (numpy, scipy and matplotlib are preinstalled;
   pytest sometimes is not):

   ```
   %pip install -r requirements.txt
   ```

4. Work as usual: `!pytest tests/test_counting.py`, and run experiments with
   `!python experiments/week01_counting_noise.py` — figures land in
   `figures/` **on your Drive**, so they persist between sessions. Edit the
   scaffold files either in Colab's file browser (double-click a `.py` file)
   or locally with Drive sync.

> **Troubleshooting**
>
> * `pip: command not found` → try `pip3`, or `python3 -m pip install -r
>   requirements.txt` (the most reliable spelling on any machine).
> * `error: externally-managed-environment` (newer Linux/Homebrew) → use the
>   venv from Option A; that is exactly what it is for.
> * numpy or scipy fails to build from source (old pip on a new Python) →
>   upgrade pip first: `python3 -m pip install --upgrade pip`, then retry;
>   recent pip downloads prebuilt wheels and never compiles.
> * matplotlib "cannot connect to display" over SSH → harmless for the
>   course: experiments never open windows, they only save PNGs.
> * `ModuleNotFoundError: No module named 'celldetect'` → you are not in the
>   `python/` directory; `cd` there and re-run.
> * `ModuleNotFoundError: No module named '_common'` → you ran an experiment
>   from inside `experiments/`. Run it from `python/`:
>   `python experiments/week01_counting_noise.py`.
> * Tests pass locally but a fresh clone fails → you probably edited the
>   solutions instead of `python/celldetect/`. Student code lives in
>   `python/celldetect/`; `python/solutions/` is the reference.
> * An experiment raises `NotImplementedError` → that week's functions are
>   not written yet. That is the expected state, not a broken repo.
