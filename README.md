# News Classification Project

This project focuses on classifying news articles into two categories: fake news and true news, using various machine learning models and text vectorization techniques. The goal is to build a robust text classification model that can accurately distinguish between fake and true news.

## Vectorization Techniques

Three different text vectorization techniques have been used:

1. **Count Vectorizer**: This technique converts text data into a numerical format based on the frequency of words.

2. **TF-IDF Vectorizer**: TF-IDF (Term Frequency-Inverse Document Frequency) is used to represent the importance of words in a document relative to the entire corpus.

3. **Word2Vec Vectorizer**: This method creates word embeddings by learning word associations within the text data.

## Machine Learning Models

The following machine learning models have been implemented and evaluated for news classification:

1. **Logistic Regression**: Implemented with variations for Count Vectorizer, TF-IDF Vectorizer, and Word2Vec Vectorizer.

2. **Naive Bayes**: Implemented with variations for Count Vectorizer, TF-IDF Vectorizer, and Word2Vec Vectorizer.

3. **Support Vector Machine (SVM)**: Implemented with variations for Count Vectorizer, TF-IDF Vectorizer, and Word2Vec Vectorizer.

4. **Random Forest**: Initially implemented with variations for Count Vectorizer, TF-IDF Vectorizer, and Word2Vec Vectorizer. An improved version of the Random Forest model with TF-IDF Vectorization is also presented.

## Evaluation

The models have been evaluated using the test dataset, and the following metrics have been calculated:

- Accuracy
- F1 Score

## Benchmark Improvement

The Random Forest model with TF-IDF Vectorization has been improved by adjusting its parameters for better performance.

## Dependencies

Ensure you have the following Python libraries installed:

- numpy
- pandas
- scikit-learn
- gensim
- matplotlib
- seaborn

## Usage

1. Clone this repository to your local machine:

```bash
git clone https://github.com/NikosMav/AI-FakeNews-Classification.git
```

2. Install the required dependencies:

```bash
pip install numpy pandas scikit-learn gensim matplotlib seaborn
```

3. Run the Jupyter Notebook or Python script to execute the machine learning models and perform news classification.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

Feel free to explore the Jupyter Notebook for a detailed step-by-step explanation of the project and its implementation.
