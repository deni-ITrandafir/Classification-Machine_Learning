Project Description
This project focuses on classifying images of cars into three categories: Audi, BMW, and Mercedes. The dataset is divided into three subsets: training, validation, and testing. Each subset contains images labeled according to the car brand they belong to.

The classification task is implemented using three machine learning models: Random Forest, k-Nearest Neighbors (k-NN), and Naive Bayes. Additionally, an ensemble model (Voting Classifier) is created to combine the strengths of the individual models through majority voting.

The project involves the following steps:

Image Preprocessing: Images are loaded, resized to 255x255 pixels, and normalized to improve model performance and computational efficiency.

Model Training: Each model is trained on the preprocessed training data, with hyperparameters chosen based on validation set performance.

Evaluation: The models are evaluated on the test set using accuracy, classification reports, and confusion matrices. A visualization of correctly and incorrectly classified images is provided to better understand model behavior.

The primary goal is to identify the most accurate and reliable approach for car classification while analyzing the strengths and weaknesses of each model. This project demonstrates the practical application of machine learning techniques in image classification tasks.
