# AI-Powered Code Optimizer 🚀 

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Model: Mistral-7B](https://img.shields.io/badge/Model-Mistral--7B-orange.svg)](https://huggingface.co/mistralai/Mistral-7B-v0.1)

An innovative tool that leverages Large Language Models (LLMs) to automatically analyze and optimize Python code. By fine-tuning **Mistral-7B** on a specialized dataset, this project provides developers with high-quality suggestions for performance enhancement and code clarity.

## 🌟 Key Features

- **LLM-Powered Optimization**: Uses a fine-tuned Mistral-7B model to generate optimized Python code.
- **Efficient Fine-Tuning**: Implements **PEFT (LoRA)** to adapt the model with minimal computational overhead.
- **Quantized Inference**: Utilizes **4-bit quantization** (bitsandbytes) for fast and memory-efficient execution.
- **Interactive UI**: A sleek **Streamlit** interface for real-time code optimization.
- **Modular Design**: Clean, professional directory structure for easy extension and maintenance.

## 📁 Project Structure

```text
Code-Optimizer/
├── assets/                 # Media assets and project presentations
├── data/                   # Training datasets
├── docs/                   # Detailed documentation and guides
├── notebooks/              # Jupyter notebooks for experimentation
├── src/                    # Source code
│   ├── app/                # Streamlit application logic
│   └── training/           # Model fine-tuning scripts
├── tests/                  # Unit tests
├── LICENSE                 # MIT License
├── README.md               # Project overview
└── requirements.txt        # Project dependencies
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA-enabled GPU (recommended for training and inference)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/SeifEldenOsama/Code-Optimizer.git
   cd Code-Optimizer
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### Running the Web Application
To start the Streamlit interface:
```bash
streamlit run src/app/application.py
```

#### Fine-Tuning the Model
To initiate the fine-tuning process:
```bash
python src/training/fine-tuning.py
```

## 📊 Methodology

The project follows a rigorous AI development lifecycle:
1. **Data Preparation**: Cleaning and formatting code pairs for supervised fine-tuning.
2. **Model Adaptation**: Applying Low-Rank Adaptation (LoRA) to the Mistral-7B base model.
3. **Quantization**: Compressing the model to 4-bit for efficient deployment.
4. **Evaluation**: Testing the model on unseen code snippets to ensure optimization quality.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Developed with ❤️ by Seif Elden Osama*
