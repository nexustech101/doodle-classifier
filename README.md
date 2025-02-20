# Doodle Classifier

A doodle classifier that allows users to create their own drawing dataset, label them into named directories, and train various machine learning models to make predictions on new doodles.

## Features
- **User Interface:** Provides a Tkinter-based GUI for drawing doodles.
- **Dataset Creation:** Users can label and save their doodles into different categories.
- **Multiple Classifiers:** Supports various classifiers including:
  - Linear SVM
  - Naive Bayes
  - Decision Tree
  - K-Nearest Neighbors
  - Random Forest
  - Logistic Regression
- **Training & Prediction:** Users can train a model on their dataset and use it for real-time predictions.
- **Model Persistence:** Models can be saved and loaded for future use.

## Installation
### Prerequisites
Ensure you have Python installed along with the required dependencies:

```sh
pip install -r requirements.txt
```

## Usage
### Running the Application
Run the following command to start the application:

```sh
python main.py
```

### How to Use
1. **Start a New Project**
   - Enter a project name.
   - Specify the number of classes (categories) for doodles.
   - Name each class.
2. **Drawing & Labeling**
   - Draw a doodle on the canvas.
   - Save it under the appropriate class label using the provided buttons.
3. **Training the Model**
   - Click the "Train Model" button to train a classifier on the collected doodles.
   - Models will be saved in the project directory that you chose in step 1.
4. **Predicting a Doodle**
   - Draw a doodle and click "Predict" to classify it.
5. **Model Management**
   - Save and load models as needed.
   - Rotate through different classifier types to test performance.

## File Structure
```
.
├── main.py           # Main application file
├── project_name/     # Directory where labeled doodles are saved
└── README.md         # Project documentation
```

## Contributing
Contributions are welcome! Feel free to submit issues or pull requests.

## License
This project is licensed under the MIT License.

