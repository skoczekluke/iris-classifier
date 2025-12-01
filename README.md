# Iris Classifier

This project trains a Decision Tree classifier on the Iris dataset and saves:
- `outputs/model.joblib`
- `outputs/confusion_matrix.png`

## How to Run (Tutor-Optimised Instructions)

1. Create and activate a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate   # Windows
   source venv/bin/activate  # Mac/Linux
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the training script:
   ```
   python src/train.py --test_size 0.2 --random_state 42
   ```

4. Check the outputs folder for:
   - model.joblib
   - confusion_matrix.png
