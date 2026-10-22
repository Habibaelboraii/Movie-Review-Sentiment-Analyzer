# 🎬 IMDB Movie Review Sentiment Analysis

An end-to-end **Natural Language Processing (NLP)** project that analyzes movie reviews and classifies them as **Positive** or **Negative** using Machine Learning and Deep Learning models.

The project explores text preprocessing, sentiment classification, model evaluation, and an interactive Streamlit web application powered by a trained Bidirectional LSTM (BiLSTM) model.

## ✨ Features

- 📝 **Text Preprocessing:** Cleans movie reviews by removing HTML tags, URLs, punctuation, and unnecessary stopwords while preserving important negation words.
- 📊 **Exploratory Data Analysis:** Examines sentiment distribution, review lengths, and text patterns.
- 🤖 **Multiple ML & DL Models:** Compares traditional Machine Learning approaches with recurrent neural networks.
- 📈 **Model Evaluation:** Uses Accuracy, Precision, Recall, F1-score, ROC-AUC, and confusion matrices where applicable.
- 🌐 **Interactive Web Application:** Analyzes custom movie reviews through a Streamlit interface.
- 🎯 **Sentiment Scores:** Displays estimated positive and negative sentiment probabilities.
- 🎨 **Modern UI:** Features a responsive layout, Dark/Light Mode, example reviews, and processed-text inspection.
- ⚠️ **Prediction Feedback:** Warns users when a review is too short or contains no recognized vocabulary words.

## 🧠 Models Implemented

| Model                              | Approach                     |
| ---------------------------------- | ---------------------------- |
| Logistic Regression + Bag of Words | Traditional Machine Learning |
| Logistic Regression + TF-IDF       | Traditional Machine Learning |
| LSTM                               | Deep Learning                |
| Bidirectional LSTM (BiLSTM)        | Deep Learning                |

The models are evaluated on a shared stratified train/test split to support a fair comparison.

## 🗂️ Dataset

The project uses the **IMDb Movie Reviews Dataset**, containing 50,000 movie reviews labeled as positive or negative.

Each review belongs to one of two sentiment classes:

- **Positive:** The review expresses a favorable opinion.
- **Negative:** The review expresses an unfavorable opinion.

Expected dataset file: `IMDB Dataset.csv`

The dataset should contain two columns:

- `review` — The original movie review.
- `sentiment` — The sentiment label.

## ⚙️ Methodology

### 1. Data Exploration

- Inspect dataset structure, missing values, and duplicate reviews.
- Analyze sentiment distribution and review lengths.
- Visualize text patterns and word distributions.

### 2. Text Preprocessing

- Remove duplicate reviews.
- Strip HTML tags and URLs.
- Normalize text to lowercase.
- Remove punctuation and non-alphabetic characters.
- Remove common English stopwords while preserving negation terms such as `not`, `no`, `nor`, and `never`.

### 3. Feature Engineering

- **Bag of Words (BoW):** Represents text using word and bigram counts.
- **TF-IDF:** Represents words using term importance weights.
- **Tokenization:** Converts text into integer sequences for neural networks.
- **Padding and Truncation:** Standardizes sequences to a maximum length of 200 tokens.

### 4. Model Training

- Train Logistic Regression models using BoW and TF-IDF features.
- Build an LSTM network for sequential text classification.
- Build a BiLSTM network to process text in both forward and backward directions.
- Apply dropout regularization and early stopping to help reduce overfitting.

### 5. Evaluation

Evaluate model performance using classification metrics and visualizations, including:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC where applicable
- Confusion matrices
- Training and validation learning curves

## 🖥️ Streamlit Web Application

The application allows users to enter an English movie review and receive a sentiment prediction.

### Application capabilities

1. Enter a custom movie review or try an example.
2. Click **Analyze Review**.
3. View the predicted sentiment.
4. Explore positive and negative probability estimates.
5. Inspect the processed review text and recognized vocabulary count.
6. Switch between Dark Mode and Light Mode.

The web application uses the saved BiLSTM model and tokenizer to process new reviews.

## 🛠️ Technologies Used





\


Additional libraries:

- NumPy
- Matplotlib
- Seaborn
- NLTK
- WordCloud

## 📁 Project Structure

```text
IMDb-Sentiment-Analysis/
│
├── app.py
├── sentiment.py
├── project2(2).ipynb
├── IMDB Dataset.csv
├── bilstm_model.keras
├── lstm_model.keras
├── tokenizer.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

**Note:** The filenames above describe the expected project organization. Keep the actual filenames consistent with your repository and the paths used in your code.

## 🚀 Installation and Usage

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd IMDb-Sentiment-Analysis
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If you do not have a requirements file yet, install the main application dependencies:

```bash
pip install streamlit tensorflow numpy nltk scikit-learn pandas matplotlib seaborn wordcloud
```

Download the English stopwords resource required by the notebook's preprocessing:

```python
import nltk
nltk.download("stopwords")
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

**Important:** The application requires `sentiment.py`, `bilstm_model.keras`, and `tokenizer.pkl` in the expected project directory. The notebook also saves `lstm_model.keras` for the LSTM model.

## 📊 Results

The notebook evaluates the classical Machine Learning baselines and the LSTM/BiLSTM models using the same test split.

| Evaluation                   | Metrics                               |
| ---------------------------- | ------------------------------------- |
| Classification performance   | Accuracy, Precision, Recall, F1-score |
| Error analysis               | Confusion Matrix                      |
| Probability-based evaluation | ROC-AUC, where applicable             |
| Training behavior            | Training and validation curves        |

*Numerical results can be added after running the final notebook and confirming the saved evaluation outputs.*

## 🔮 Future Improvements

- Compare additional neural architectures and text representations.
- Improve handling of sarcasm, ambiguous expressions, and complex negation.
- Add model comparison charts to the web interface.
- Deploy the application publicly.
- Add automated tests for preprocessing and inference.

## 👩‍💻 Author

**Habiba Saber Elboraii**

AI & Machine Learning | Deep Learning | Natural Language Processing

- GitHub: [Your GitHub Profile](https://github.com/YOUR_USERNAME)
- LinkedIn: [Your LinkedIn Profile](https://www.linkedin.com/)

---

⭐ If you find this project interesting, feel free to explore the repository and share your feedback.

