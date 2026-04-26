# 🚫 Comment Toxicity Classifier

A deep learning project designed to detect and categorize toxic comments across multiple categories such as toxicity, threats, insults, and identity hate. This system uses Natural Language Processing (NLP) techniques and a Bidirectional LSTM architecture to understand context and intent in text.

---

## 🛠 Tech Stack

- **Languge**: Python 3.10+
- **Deep Learning Framework**: TensorFlow 2.15 (Legacy Keras compatibility)
- **Data Manipulation**: Pandas, NumPy
- **Preprocessing**: Keras TextVectorization
- **Model Architecture**: Bidirectional LSTM (Long Short-Term Memory)
- **Evaluation**: Scikit-learn, Keras Metrics
- **User Interface**: Gradio (Web-based interactive GUI)
- **Visualization**: Matplotlib

---

## 📊 Dataset Details

The project utilizes the **Jigsaw Toxic Comment Classification Challenge** dataset from Kaggle.
- **Source**: Wikipedia talk page comments.
- **Labels**: 159,571 comments labeled by human raters.
- **Categories**: 6 binary labels (Toxic, Severe Toxic, Obscene, Threat, Insult, Identity Hate).
- **Class Imbalance**: Most comments are non-toxic, making the detection of rare categories (like 'Identity Hate' or 'Threat') more challenging.

---

## 🧪 Example Test Cases

You can use the following comments to test the model's sensitivity in the Gradio interface:

| Comment Type | Example Text | Expected Behavior |
| :--- | :--- | :--- |
| **Neutral** | "I disagree with this edit, but I respect your opinion." | All categories should be False. |
| **Toxic** | "You are completely wrong and you shouldn't be allowed to edit." | High probability for 'Toxic'. |
| **Threat** | "I am coming for you, you won't see another day." | High probability for 'Threat'. |
| **Obscene** | "[Expletive] you and your stupid website!" | High probability for 'Obscene' and 'Insult'. |
| **Identity Hate** | "People from your country don't belong here." | High probability for 'Identity Hate'. |

---

## 🔬 Deep Dive: Why Bidirectional LSTM?

Standard RNNs only read text from left to right. However, the toxicity of a word often depends on the words that come *after* it.
- **Standard LSTM**: "I will kill..." (seems like a threat, but might end with "...the mood")
- **Bidirectional LSTM**: Reads "I will kill the mood" from both sides, understanding that "mood" changes the intent of "kill".

---

## 🚀 How to Run the Model

The system follows a standard NLP pipeline:

1.  **Data Acquisition**: Loads the Jigsaw Toxic Comment Classification dataset containing over 150,000 labeled comments.
2.  **Preprocessing**:
    *   **Text Vectorization**: Converts raw text into integers.
    *   **Vocabulary Adaptation**: Learns the top 200,000 words from the training corpus.
    *   **Dataset Pipeline**: Uses the `tf.data` API for efficient buffering (`cache`, `shuffle`, `batch`, `prefetch`).
3.  **Model Architecture**:
    *   **Embedding Layer**: Maps words to 32-dimensional dense vectors.
    *   **Bidirectional LSTM**: Processes text in both forward and backward directions to capture full context.
    *   **Dense Layers**: Extraction of high-level features through ReLU-activated fully connected layers.
    *   **Output Layer**: 6 units with Sigmoid activation (one for each toxicity category).
4.  **Training**: Optimized using `BinaryCrossentropy` and the `Adam` optimizer.
5.  **Evaluation**: Precision, Recall, and Categorical Accuracy are measured for performance.
6.  **Deployment**: A Gradio web interface provides a real-time way for users to test the model.

---

## 🚀 How to Run the Model

### 1. Prerequisites
Ensure you have Python 3.10+ installed on your system.

### 2. Set Up Virtual Environment
It is highly recommended to use a virtual environment to manage specific dependency versions.

```powershell
# Create venv
python -m venv venv

# Activate venv (Windows)
.\venv\Scripts\activate
```

### 3. Install Dependencies
The model requires TensorFlow 2.15 to load the pre-trained weights correctly.

```bash
pip install "tensorflow<2.16" pandas gradio scikit-learn matplotlib jinja2
```

### 4. Running the Evaluation
Launch the Gradio interface:

```bash
python run_notebook.py
```

### 5. Accessing the UI
Open your browser to:
**`http://127.0.0.1:7860`**

---

## 📊 Model Categories
The model predicts:
*   **Toxic**
*   **Severe Toxic**
*   **Obscene**
*   **Threat**
*   **Insult**
*   **Identity Hate**

---

## 📂 Project Structure
*   `Toxicity.ipynb`: Original development notebook.
*   `toxicity.h5`: Pre-trained model file.
*   `run_notebook.py`: Automated execution script.
*   `jigsaw-toxic-comment-classification-challenge/`: Dataset source.
*   `venv/`: Project environment.

---
*Created for automated toxicity evaluation.*
