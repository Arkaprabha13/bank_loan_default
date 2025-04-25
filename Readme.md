# Bank Loan Default Prediction

## 📊 Project Overview

This project implements neural network models to predict the likelihood of loan defaults, helping financial institutions make informed lending decisions. By analyzing borrower characteristics and loan details, we can identify potential high-risk applicants before loan approval.

## 🎯 Problem Statement

Banking institutions face significant financial risks when loans default. Our machine learning solution helps mitigate these risks by:

- Accurately predicting the probability of loan default
- Comparing different neural network architectures
- Providing insights through comprehensive visualizations
- Supporting risk-based decision making

## 🧠 Model Architectures

We implemented and compared two neural network architectures:

### Artificial Neural Network (ANN)
- Simple yet effective architecture
- Regularization with dropout to prevent overfitting
- Binary classification with sigmoid activation

### Multi-Layer Perceptron (MLP)
- Advanced architecture with additional hidden layers
- Batch normalization and regularization techniques
- Enhanced representation capabilities for complex patterns

## 📈 Results

### Confusion Matrices

The confusion matrices show how our models performed in classifying loan defaults:

![Confusionatrices

Key observations:
- ANN achieved 44,982 true negatives and 300 true positives
- MLP achieved 44,968 true negatives and 291 true positives
- Both models show similar performance in identifying defaults

### Training Performance

The training history demonstrates model convergence and validation performance:

![Model Training History](model_training_history.jpg reached validation accuracy ~88.8%
- ANN showed slightly better validation loss
- Training was stable with no significant overfitting
- Models converged after approximately 30 epochs

### ROC Analysis

ROC curves provide insight into the classification performance across different thresholds:

ROC Curves

Results:
- ANN achieved an AUC of 0.758
- MLP achieved a marginally better AUC of 0.759
- Both models significantly outperform random classification (diagonal line)

## 💻 Installation & Usage

```bash
# Clone the repository
git clone https://github.com/yourusername/loan-default-prediction.git
cd loan-default-prediction

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the training script
python train_models.py

# Run the prediction application
streamlit run app.py
```

## 🛠️ Technologies Used

- Python 3.8+
- TensorFlow/Keras for neural network implementation
- Scikit-learn for preprocessing and evaluation
- Pandas for data manipulation
- Matplotlib and Seaborn for visualization
- Streamlit for the interactive web application

## 🔄 Model Deployment

The models are deployed through a Streamlit application that allows users to:
- Input borrower information
- Receive real-time default risk predictions
- Compare results between different models
- Visualize prediction explanations

## 🔍 Future Improvements

- Implement more sophisticated architectures (LSTM, Transformers)
- Address class imbalance with advanced sampling techniques
- Incorporate feature importance visualization
- Develop ensemble methods to combine model strengths
- Integrate with banking systems through API endpoints

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Contributors

- Your Name - [GitHub Profile](https://github.com/yourusername)

## 📧 Contact

For questions or feedback about this project, please contact [your-email@example.com](mailto:your-email@example.com).

Citations:
[1] https://pplx-res.cloudinary.com/image/private/user_uploads/dYDylFAPbdaflbR/confusion_matrices.jpg
[2] https://pplx-res.cloudinary.com/image/private/user_uploads/ovolzarpKvdNvIq/model_training_history.jpg
[3] https://pplx-res.cloudinary.com/image/private/user_uploads/crbVZeDfLqGlDaM/roc_curves.jpg

---
Answer from Perplexity: pplx.ai/share
