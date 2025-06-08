# 💬 Emotion-Aware Chatbot

A Streamlit-based chatbot that detects emotions from user input and offers supportive responses like quotes, journaling prompts, and music.

---

## 📊 Dataset Used
- A labeled emotion dataset with text samples classified into **happy**, **sad**, **angry**, and **fear**.
- Used for training a supervised machine learning model.

---

## 🧠 Approach Summary
- **Preprocessing**:
  - Cleaned and vectorized input text using **TF-IDF** to convert it into numerical features.
- **Model Training**:
  - Trained a **Logistic Regression** classifier to predict emotional labels based on the text.
  - Evaluated using accuracy and confusion matrix (see Jupyter notebook).
- **Deployment**:
  - Saved the model and vectorizer with `joblib`.
  - Built a frontend using **Streamlit** where:
    - Users input how they’re feeling in a text area.
    - The app predicts the emotion using the trained model.
    - Based on the emotion, it displays a motivational **quote**, a **journaling prompt**, or a **YouTube music link** using embedded video.
- **Design**:
  - The app uses a custom pastel-themed UI with a clean layout and sidebar for user guidance.

---

## 📦 Dependencies

```bash
streamlit
pandas
scikit-learn
joblib
