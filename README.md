# Movie Genre Classification

## Overview
This project focuses on classifying movie genres using machine learning techniques. The primary objective is to predict movie genres based on a combination of textual, numerical, and categorical data, such as overviews, runtime, and keywords. The project is divided into five distinct phases, each contributing to the development of a robust multi-label classification model and a utility application.

---

## Motivation
The volume of movies on streaming platforms makes manual genre classification impractical. Machine learning provides a scalable, efficient, and accurate solution for classifying movie genres, aiding in content management, recommendation systems, and user experience optimization.

---

## Dataset
### Source
- **TMDb Movies Dataset (2023)** from Kaggle
  - Approximately 930,000 entries
  - 24 features, with a subset selected for this project.

### Key Features:
- **Textual**: Overview, keywords
- **Numerical**: Runtime, vote average, vote count
- **Categorical**: Genres, original language, status

### Preprocessing Steps:
- Missing values handled, irrelevant features removed.
- Text features cleaned (stop word removal, stemming, punctuation removal).
- Numerical features scaled using Min-Max Scaling and Standard Scaling.
- One-hot encoding applied to genres for multi-label classification.

---

## Project Phases

### Phase I: Problem Identification and Literature Review
- Identified movie genre classification as a multi-label classification problem.
- Reviewed existing approaches, focusing on models like CNNs and LSTMs.
- Chose to use natural language processing (NLP) and simpler models like decision trees and random forests due to project constraints.

### Phase II: Data Preparation and Feature Engineering
- Cleaned and preprocessed the dataset to ensure consistency.
- Applied feature engineering, including:
  - One-hot encoding for genres.
  - Scaling for numerical features.
  - Initial TF-IDF transformation for textual features (later replaced).

### Phase III: Pilot Study
- Evaluated six supervised learning models:
  1. Decision Trees
  2. K-Nearest Neighbors (KNN)
  3. Logistic Regression
  4. Naive Bayes
  5. Neural Networks
  6. Random Forests
- Replaced TF-IDF with Word2Vec embeddings to improve the representation of textual features.
- Neural Networks emerged as the top choice due to their ability to handle text embeddings and non-linear relationships.

### Phase IV: Model Design and Optimization
- Designed a Neural Network architecture with:
  - Three hidden layers (512, 256, 128 units) with ReLU activations.
  - Dropout for regularization.
  - Sigmoid output layer for multi-label classification.
- Implemented optimization techniques:
  - Focal loss to address class imbalance.
  - Manual, grid, and random search for hyperparameter tuning.
  - Early stopping and learning rate scheduling.

### Phase V: Model Implementation and Utility Application
- **Model**:
  - Trained using the finalized architecture and focal loss.
  - Integrated hyperparameter tuning for improved performance on underrepresented genres.
- **Utility Application**:
  - Designed a client-server architecture.
    - **Client**: Handles user interaction and basic preprocessing.
    - **Server**: Executes advanced preprocessing, model inference, and data storage.
  - RESTful API with endpoints for classification and history retrieval.
  - GUI developed using Tkinter for user-friendly interaction.

---

## Application Features
1. **Genre Classification**
   - Predict genres based on user-provided movie metadata (e.g., title, description).
2. **Feedback Collection**
   - Users can provide feedback to improve model accuracy.
3. **History Tracking**
   - Maintains a record of past classifications for review and analysis.
4. **Online Model Retraining**
   - Incorporates new data and feedback into the model to adapt to emerging trends.

---

## Model Evaluation
Metrics used to evaluate the model:
- Precision, Recall
- Macro, Micro, and Weighted F1-scores

Best-performing model:
- **Neural Network**:
  - High weighted precision and recall.
  - Superior performance with textual embeddings.

---

## Technical Stack
- **Languages**: Python
- **Libraries**: TensorFlow, Keras, Scikit-learn, Flask, Tkinter
- **Tools**: Kaggle, SQLite, Word2Vec

---

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/FaridaBey/MovieGenreClassifier_ML.git
   cd movie-genre-classification
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements
   ```
3. Start the server:
   ```bash
   python server.py
   ```
4. Launch the client application:
   ```bash
   python client.py
   ```

---

## Future Work
- Expand dataset to include more diverse genres.
- Integrate advanced text preprocessing techniques.
- Develop a web-based interface for broader accessibility.

---

## Contributors
- **Farida Bey** 
- **Youssef Ghaleb** 

---

## References
1. Marina Ivasic-Kos et al., *Automatic Movie Posters Classification into Genres*, 2015.
2. IEEE Conference Publication, *Movie Genre Classification with Convolutional Neural Networks*, 2016.
3. Kaggle Datasets: TMDb Movies Dataset 2023, Genre Classification IMDb.

