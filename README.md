# IMDb Movie Review Sentiment Analysis

A complete deep learning project for classifying English movie reviews as **positive** or **negative**. It combines a notebook based NLP workflow with a ready to run Streamlit app powered by a trained Bidirectional LSTM (BiLSTM).

> Type or paste a review into the app to see its predicted sentiment, positive and negative scores, and the text produced by preprocessing.


## Project overview

The project explores sentiment classification on the IMDb movie review dataset. The notebook performs data inspection and exploratory analysis, cleans the review text, compares classical machine learning baselines with recurrent neural networks, and saves the selected BiLSTM model and tokenizer. The Streamlit app uses those saved artifacts to predict sentiment for new reviews.

**Task:** Binary text classification  
**Input:** English movie review text  
**Output:** Positive or negative sentiment and a model score  
**Interface:** Streamlit  
**Deep learning framework:** TensorFlow / Keras

## How it works

### 1. Data preparation

The notebook reads `IMDB Dataset.csv`, which is expected to contain `review` and `sentiment` columns. It removes duplicate rows and prepares the text and sentiment labels for modeling. A stratified split reserves 20% of the data for testing, with a fixed random seed (`42`).

Text preprocessing removes HTML tags, URLs, punctuation, and non-English letters, converts text to lowercase, and removes common English stopwords. The negation words `no`, `not`, `nor`, and `never` are retained because they can change the meaning of a review.

### 2. Models explored

The notebook compares four approaches:

| Model | Text representation | Purpose |
| --- | --- | --- |
| Logistic Regression | Bag-of-Words counts | Classical baseline |
| Logistic Regression | TF-IDF weights | Classical baseline that downweights common terms |
| LSTM | Token sequences | Learns word order and context |
| BiLSTM | Token sequences | Reads sequences in both directions; this is the model used by the app |

For the classical models, the notebook uses up to 30,000 unigram and bigram features. For the neural models, a Keras tokenizer keeps a vocabulary of up to 10,000 words, and reviews are padded or truncated to 200 tokens. Both recurrent models use 128-dimensional embeddings and a sigmoid output for binary classification. Training uses Adam, binary cross-entropy, a batch size of 64, up to 10 epochs, and a validation split with early stopping.

### 3. Prediction in the app

The app cleans the entered review with the inference helper in `sentiment.py`, converts it to a token sequence, pads it to the model's expected length, then passes it to `bilstm_model.keras`. A score of 0.5 or higher is labeled positive; a lower score is labeled negative. The app also shows the complementary score for the other class and warns when a review is very short or contains no words recognized by the tokenizer.

## Web app

The Streamlit interface includes:

- A review form and sample positive and negative reviews.
- Positive/negative classification with both class scores.
- A view of the cleaned text and recognized vocabulary count.
- Dark and light display modes.
- Short-review and unknown-vocabulary guidance.

## Repository files

| Path | Description |
| --- | --- |
| `app.py` | Streamlit interface, asset loading, and prediction display |
| `sentiment.py` | Shared review cleaning and BiLSTM inference functions |
| `bilstm_model.keras` | Trained BiLSTM model loaded by the app |
| `tokenizer.pkl` | Fitted Keras tokenizer used to encode reviews |
| `IMDB Dataset.csv` | Dataset used by the notebooks |
| `project2.ipynb` | Recommended notebook with the reproducible analysis and model comparison workflow |
| `prohect2.ipynb` | Earlier notebook version with saved outputs |
| `requirements.txt` | Runtime dependencies for the Streamlit app |

## Run the app

Use Python 3.10 to 3.12 and run the following from the repository root. The app requires the model and tokenizer files to remain beside `app.py`.

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

## Run the notebooks

The root `requirements.txt` is intentionally for the app. To run `project2.ipynb`, first install Jupyter and the notebook's analysis libraries in the active environment:

```bash
python -m pip install jupyter pandas scikit-learn nltk matplotlib seaborn wordcloud
```

Download the NLTK English stopwords corpus once:

```bash
python -c "import nltk; nltk.download('stopwords')"
```

Start Jupyter from the repository root and open `project2.ipynb`:

```bash
python -m jupyter notebook
```

Keep `IMDB Dataset.csv` beside the notebook, or change `DATA_PATH` in its data loading cell. The notebook trains models and can regenerate the saved model and tokenizer artifacts.

## Model evaluation

The notebook evaluates the models on the same held-out test set. It reports accuracy, precision, recall, and F1 score; it also explores ROC-AUC, ROC curves, confusion matrices, and classification reports. Results depend on the data split, preprocessing, and training environment, so run the notebook to obtain the metrics for your current execution rather than assuming a fixed score.

## Limitations

- The model is intended for English movie reviews; results for other languages and other kinds of text may be unreliable.
- Very short reviews provide little context. The app flags these cases, but still displays the model's estimate.
- Scores are model outputs, not calibrated probabilities or guarantees of a review's true sentiment.
- The model and tokenizer must come from the same training run because their vocabulary mappings need to match.

## Data and large files

The project uses the [IMDb Dataset of 50K Movie Reviews on Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). Download the dataset from Kaggle and place `IMDB Dataset.csv` in the repository root, beside the notebooks. The notebook expects the CSV to contain `review` and `sentiment` columns.

The CSV is approximately 66 MB, above GitHub's 50 MB file warning threshold. If you plan to commit it, use [Git LFS](https://git-lfs.com/); alternatively, leave it out of the repository and download it from Kaggle when you want to run the notebooks. The Streamlit app does not need the CSV at runtime: it uses only `bilstm_model.keras` and `tokenizer.pkl`.

Before publishing, check that you have the right to redistribute the dataset and any model artifacts included in the repository.
