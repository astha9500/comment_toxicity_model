# 🚫 Comment Toxicity Classifier: NLP for Online Safety

![Project Banner](assets/banner.png)

A high-performance Deep Learning solution designed to detect and categorize toxic comments into six distinct categories. Built using a **Bidirectional LSTM** architecture, this project aims to foster healthier online communities by identifying harmful language with high precision.

---

## ✨ Key Features

*   **Multi-Label Classification**: Simultaneously detects Toxicity, Severe Toxicity, Obscene language, Threats, Insults, and Identity Hate.
*   **Context-Aware Analysis**: Utilizes Bidirectional LSTMs to understand the nuances of language from both directions.
*   **Real-Time GUI**: Includes a sleek **Gradio** web interface for instant model testing.
*   **Efficient Pipeline**: Implements the `tf.data` API for lightning-fast data processing and buffering.
*   **Production Ready**: Model weights are saved and ready for deployment.

---

## 📸 Project Preview

| Interactive Interface | Classification Results |
| :---: | :---: |
| ![Interface Demo](assets/demo.gif) | ![Results Graph](assets/results.png) |

---

## 🛠 Tech Stack

*   **Language**: Python 3.10+
*   **Deep Learning**: TensorFlow 2.15 (Bidirectional LSTM)
*   **Data Pipeline**: Pandas, NumPy, Keras TextVectorization
*   **UI/UX**: Gradio
*   **Visualization**: Matplotlib
*   **Evaluation**: Scikit-Learn Metrics

---

## 🧠 Model Architecture

The core of this project is a **Bidirectional Long Short-Term Memory (Bi-LSTM)** network. Unlike traditional RNNs, Bi-LSTMs process text sequences in both directions (past to future and future to past), allowing the model to capture the full context of a word based on its surroundings.

1.  **Embedding Layer**: Converts words into 32-dimensional dense vectors.
2.  **Bidirectional LSTM**: Core processing unit with 64 units per direction.
3.  **Feature Extraction**: Dense layers with ReLU activation for high-level pattern recognition.
4.  **Output Layer**: 6 units with Sigmoid activation for independent multi-label probabilities.

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and install the dependencies. It is recommended to use a virtual environment.

```bash
# Clone the repo
git clone https://github.com/astha9500/comment_toxicity_model.git
cd comment_toxicity_model

# Setup venv
python -m venv venv
.\venv\Scripts\activate

# Install requirements
pip install "tensorflow<2.16" pandas gradio scikit-learn matplotlib
```

### 2. Usage
To launch the interactive Gradio interface:

```bash
python run_notebook.py
```

Open your browser and navigate to `http://127.0.0.1:7860` to start testing comments.

---

## 📊 Dataset Reference
This project uses the **Jigsaw Toxic Comment Classification** dataset from Kaggle, containing over 150k labeled comments from Wikipedia talk pages.

---

## 🤝 Contributing
Contributions are welcome! If you have suggestions for improving model accuracy or adding new features, feel free to open an issue or submit a pull request.

---

*Built with ❤️ for a safer internet.*
