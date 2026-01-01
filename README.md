# AI-Powered Code Optimizer: Enhancing Python with LLMs

## Project Overview

The **Code Optimizer** is an innovative project that leverages the power of Large Language Models (LLMs) to automatically analyze and optimize Python code. The core of this system involves fine-tuning a state-of-the-art language model, **Mistral-7B**, on a dedicated dataset of code optimization examples. The goal is to provide developers with a tool that can suggest and implement performance-enhancing changes to their code, thereby improving efficiency and execution speed.

This repository contains the complete workflow, from data preparation and model fine-tuning to the deployment of a user-friendly Streamlit application.

## Key Features

The Code Optimizer project delivers the following capabilities:

*   **LLM-Powered Optimization:** Utilizes a fine-tuned **Mistral-7B** model to generate optimized versions of input Python code.
*   **Efficient Fine-Tuning:** Implements **Parameter-Efficient Fine-Tuning (PEFT)** using the **LoRA** (Low-Rank Adaptation) technique, significantly reducing computational resources and time required for training.
*   **Quantized Inference:** Employs **4-bit quantization** via the `bitsandbytes` library to enable efficient model loading and faster inference on resource-constrained environments.
*   **Interactive Web Application:** A user-friendly interface built with **Streamlit** allows users to input their code and receive the optimized output in real-time.
*   **Deployment Ready:** The application script includes integration with **Pyngrok** for easy public exposure and testing of the Streamlit application.

## Technology Stack

The project is built upon a robust set of modern AI and Python libraries:

| Category | Technology | Purpose |
| :--- | :--- | :--- |
| **Model** | Mistral-7B-v0.1 | The base Large Language Model used for code generation. |
| **Fine-Tuning** | PEFT (LoRA) | Efficiently adapts the base model to the code optimization task. |
| **Framework** | Hugging Face `transformers` | Provides the necessary tools for loading and interacting with the LLM. |
| **Deployment** | Streamlit | Creates the interactive web interface for the application. |
| **Efficiency** | `bitsandbytes`, `accelerate` | Handles 4-bit quantization and distributed training/inference. |
| **Networking** | Pyngrok | Exposes the local Streamlit application to the public internet for sharing. |
| **Language** | Python | The primary programming language for the entire project. |

## Project Structure

The repository is organized into three main directories:

```
.
├── code/
│   ├── analysis.ipynb          # Notebook for initial data exploration and analysis.
│   ├── application.ipynb       # Notebook version of the Streamlit application logic.
│   ├── application.py          # The main Streamlit application script for inference.
│   ├── fine-tuning.ipynb       # Notebook detailing the fine-tuning process (PEFT/LoRA).
│   └── fine-tuning.py          # Python script for running the fine-tuning process.
├── data/
│   └── code_optimization_dataset.csv # The dataset used for fine-tuning the LLM.
└── presentation/
    └── AI-Powered-Code-Optimizer-Enhancing-Python-with-LLMs.pdf # Project presentation/report.
```

## Installation and Setup

To set up the project locally, follow these steps:

### Prerequisites

Ensure you have Python 3.x installed and a system capable of handling large models (GPU recommended for fine-tuning and efficient inference).

### 1. Clone the Repository

```bash
git clone https://github.com/SeifEldenOsama/Code-Optimizer.git
cd Code-Optimizer
```

### 2. Install Dependencies

The project relies on several libraries for model handling and the web interface.

```bash
pip install -r requirements.txt
# Note: A requirements.txt file is assumed based on the code. 
# The necessary packages include: transformers, peft, accelerate, streamlit, pyngrok, sentencepiece, bitsandbytes, torch
```

### 3. Model Fine-Tuning (Optional)

If you wish to reproduce the fine-tuning process, refer to the scripts in the `code/` directory. This step requires a powerful GPU and the dataset located in `data/`.

```bash
# Example command to run the fine-tuning script
python code/fine-tuning.py
```

### 4. Running the Application

The core application is a Streamlit app. You will need to set your `NGROK_AUTH_TOKEN` environment variable to use the Pyngrok feature for public access.

1.  **Set Environment Variable:**
    ```bash
    export NGROK_AUTH_TOKEN="YOUR_NGROK_AUTH_TOKEN"
    ```
2.  **Run the Streamlit App:**
    ```bash
    streamlit run code/application.py
    ```

The application will start, and if the `NGROK_AUTH_TOKEN` is set, it will provide a public URL to access the Code Optimizer web interface.

## Usage

1.  Open the Streamlit application in your browser.
2.  Enter the Python code you wish to optimize into the provided text area.
3.  Click the **"Generate"** button.
4.  The fine-tuned Mistral model will process the input and display the optimized code snippet.

---
