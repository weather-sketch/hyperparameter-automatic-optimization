# hyperparameter-automatic-optimization

## Introduction
The investigation aimes to explore hyperparameter optimization on Wine Quality Dataset. I will follow Chollet's universal workflow of Deep Learning With Python. The overall goal of this project is to develop a program that systematically explores the hyperparameter space to identify the most effective configuration for the model. This involves implementing a search strategy, in this case, random search, to sample different hyperparameter combinations from a predefined space. IWe will assess the model's performance using cross-validation, a robust statistical method that maximizes the use of available data by iteratively splitting the dataset into training and validation subsets. This practice helps in estimating the performance of the model on unseen data, thereby ensuring that our optimization process generalizes well and does not overfit to particular quirks of the training data.

By the end of this investigatiI aimpire to have a model tuned to offer the highest predictive accuracy on the Wine Quality Dataset. This report will detail the steps taken in the universal workflow, the implementation of our hyperparameter optimization program, the results obtained, and a discussion of the outcome.

## Methodology
### Step 1: Define the Problem and Assemble a Dataset
The Wine Quality dataset typically refers to one of two datasets available in the UCI Machine Learning Repository related to red and white variants of the Portuguese "Vinho Verde" wine. Wine quality is a subjective variable, judged by experts, and scored on a scale from 0 (very bad) to 10 (excellent). The red wine dataset contains 1,597 instance while the white wine dataset contains 4,898 instancs. I will mainly use the red wine dataset for more quicker training.

Each dataset has **11 input features** :
Fixed acidity,
Volatile acidity,
Citric acid,
Residual sugar,
Chlorides,
Free sulfur dioxide,
Total sulfur dioxide,
Density,
pH,
Sulphates,
Alcohol.

The output variable is the quality of the wine, scored on a scale from 0 to 10. Therefore, it can be treated as either a regression problem if we consider the output variable as continuous, or a classification problem for a discrete output.

For the purpose of this project, I converted it into a classification problem where each wine is great(a score of 7 or higher), good(a score between 4 and 6), not good(a score below 4).

In this way, it is transformed into a multi-class classification problem, and my goal is to classify the red wine into the right category. After identifying the problem type, the next step is to determine the choice of model architecture and evaluation metric.

### Step 2: Choose a Evaluation Metric
For multi-class classfication problems, categorical entropy and accuracy would be the most frequently-used metrics. In this case, given input feature, the model should be able to classified the data into the right category.

### Step 3: Decide on a Evaluation Protocol
Evaluation Protocol is essential to keep track of the progress as I tune my models. There are three common ways:
Hold-out validation set
K-fold cross-validation
Iterated K-fold validation

I will use both K-fold cross validation and Hold-out validation set as evaluation protocol. The holdout test set will be used to assess the model's final performance after tuning and validating the model through K-fold cross-validation on the training set. Wine Quality dataset is small dataset, where overfitting can be a concern. Cross-validacan be one of the solution to ensure the model generalizes well to unseen data.

For smaller datasets, each fold in K-fold cross-validation contains less data, which means the validation scores might have higher variance. In such cases, K-fold cross-validation is indeed very suitable because it ensures that each observation is used for both training and validation exactly once, making the most out of limited data.

I have an exact quantity of 1597 pieces of data. I am not sure what is the most appropriate number for k, might be 3, 5, or 10. I will conduct experimentations on this to see if there is any differences.

### Step 4: Prepare the Data
Now I have determined the dataset, the evaluation metrics, the evaluation protocol. It is time to prepare the data and preprocess them.

- Missing Values

Missing values in the dataset are being filled with the mean value of their respective columns. This is a common imputation technique to handle misg data.

- Extract Feature and Target Value

Input features are being taken from all columns except the last two, and target variable is taken from the second last column of he DataFram. The .to_numpy() method is used to convert the DataFrame slices into numpy arrays.

- Standardize Input Features

The input features are being standardIzed using StandardScaler, which removes the mean and scales the data this way. It is an essential preprdocessing step to normalize the data before feeding it to machine learning model.

The target variable y is being relabelled to three classes: Class 0: for original value between 6 and 9 (inclusive) Class 1: for original value between 4 and 6 (inclusive) Class 2: for all other values

- One-hot Encode the Target

The relabeled target variable is one-hot encoded, which is necessary for multi-class classification with neural networks. This creates matrix representation of the data.

- Split Data into Training and Test Sets

The scaled data is split into training and test sets using an 80-20 split. 80% of the data is used for training, and 20% is reserved for testing the model's performance. The random_state parameter ensures reproducibility of the split.

- Set Up K-Fold Cross-Validation

A KFold cross-validator is set up to use 3 folds first, as discussed.

### Step 5: Build a Basic Model

I will build a neural network between features and the target. Key hyperparameters for tuning included the architecture of the neural network (number of layers and units), learning rate, optimizer type, and regularization parameters. Random search was chosen to efficiently navigate the hyperparameter space with a predefined range for each parameter. And in later experiment, I will try to include more hyperparameters.
![image](https://github.com/weather-sketch/hyperparameter-automatic-optimization/assets/138662766/c0c6f725-fd8b-4834-9bdd-7af7efb1e940)

**Experiment 1 Result Analysis**
The provided best score indicates a good level of accuracy; however, it shows a sign of overfitting. T If the model's accuracy on the training santly higher than its accuracy on the validatio.nte mmy's ability to geneI will

In your next experiment, you can implement batch normalization by addmodeltion layers i. It is normallyural neDenseure, typand y after fully connected layers buAnd I will functions). It is also advisable to monitor both training and validation metrics closely to determine if batch normalization helps in reducing the gap between training and validation performance, which would be a good indicator of mitigated overfitting.

### Step 6: Scale up — Experiment 2: Kfold = 10, batch_normalization
In the next experiment, I will add batch normalization function.

Batch Normalization: True, False
Batch normalization is a technique to provide any layer in a neural network with inputs that are zero mean/unit variance, which helps to stabilize the training process.

**Experiment 2 Result Analysis**
The integration of batch normalization into the training process seems to have had a nuanced effect on the model's performance.

No Significant Improvement on Accuracy The inclusion of batch normalization did not lead to a notable increase in accuracy. This might suggest that the model’s ability to fit the data was not primarily constrained by internal covariate shift or that other factors are limiting the model's performance.

Mitigating Overfitting The observation that signs of overfitting have been reduced is encouraging. Batch normalization can have a regularizing effect, as it introduces a small amount of noise into each layer's inputs during training. This can prevent the model from fitting too closely to the training data.

The fact that batch normalization helped mitigate overfitting but did not improve accuracy might indicate that the model is regularized enough not to overfit but still lacks the capacity to better learn the underlying patterns in the data. This could also be a result of other hyperparameters or aspects of the model architecture that are not optimal.

For the next steps, I will include weight initialization. Proper weight initialization can have an impact on the training dynamics of a neural network.

## Step 7: Regularizing the Model — Experiment 3: Kfold = 3, batch_normalization, weight initialization
Batch Normalization: True, False
Batch normalization is a technique to provide any layer in a neural network with inputs that are zero mean/unit variance, which helps to stabilize the training process.

Weight Initializers: random_normal, random_uniform, he_normal, he_uniform, glorot_normal, glorot_uniform
Weight initialization hyperparameters control how the initial weights of a network are set. Proper initialization can improve convergence during training.

**Experiment 3 Result Analysis**
The inclusion of weight initialization in the training process seems to have had an interesting, yet not entirely positive, impact on model performance.

Still No Significant Improvement on Accuracy

Adding weight initialization did not lead to a substantial increase in model accuracy. This could suggest the model's performance could be bounded by other factors, such as the need for a more sophisticated model architecture or more representative training data.

Best Performance Peaked Early

I observed that the model's best performance occurred before the end of training is indicative of the fact that at some point, further training leads to overfitting or is no longer beneficial. By including early stopping in the training process, halt training when the model's performance on the validation set begins to degrade, would be helpful prevent overfitting.

K-Fold Cross-Validation Variation

In this case, varying the number of folds in K-Fold cross-validation didn't make much difference. This is might be a good sign, suggesting that the model is not sensitive to the specific splits of the data. I will choose to use 3 folds for cross-validation is a practical decision that will reduce computational load.

### Experiment 4: Kfold = 3, batch_normalization, weight initialization, early-stopping

**Experiment 4 Result Analysis**
Including early stopping in the training process and finally saw an improvement in results.

Early Stopping

By incorporating early stopping, the training process was halted before the model could overfit to the training data, thereby improving its generalization to unseen data. Additionally, early stopping helped in identifying the optimal number of epochs to train, which is when the model achieves the best performance on the validation dataset.

### Finalize the Model
Now that training process has been outlined that includes a mechanism to prevent overfitting and to stop at the peak of the model's performance, it is time to finalize the model. Here are the next steps:

Finalize Hyperparameters: The last set of hyperparameters are associated with the best validation performance during tuning.

Re-train Model: Use these hyperparameters to re-train the model on the full training dataset.

Performance Evaluation: After re-training, evaluate the model on the held-out test set to confirm that the improvements hold on completely unseen data.
![image](https://github.com/weather-sketch/hyperparameter-automatic-optimization/assets/138662766/dfa9bce4-64b7-4ebd-b19f-25185edacabe)

## Limitation and Discussion
This investigation represents a comprehensive examination of hyperparameter optimization, encompassing an extensive range of tunable parameters. It has facilitated a more profound comprehension of the roles hyperparameters play within the training process. The exploration began with a critical assessment of the parameter space, marking the first occasion where serious consideration was given to the impact of varying ranges. Employing both hold-out and K-fold cross-validation concurrently offered valuable insights into their operational mechanics and their advantages, particularly for smaller datasets.

Nonetheless, limitation is obviousints. The peak accuracy attained by the optimal model configuration is approximately 88.2%, which, while substantial, is ideal enoughlary. The potential reasons for this are multifaceted:

Dataset Size: In an effort to expedite the training cycle and conserve computational resources, a smaller dataset was deliberately chosen. This decision likely impeded the model's capacity for further refinement and evolution.
Impact of Data Volume: The modest scale of the dataset may have also restricted the full potential of the experimental procedures.

Future Directions: The progression of this research has culminated in the establishment of a solid foundational model. Prospective explorations could encompass:

Further Hyperparameter Tuning: Additional adjustments and tuning of hyperparameters may yield further improvements.
Ensemble Techniques: Integrating multiple models might enhance overall performance.
Feature Engineering: The development or enhancement of features could significantly bolster the model's ability to discern more complex data patterns.

In conclusion, this investigation stands as a successful conceptual proof of hyperparameter optimization's efficacy.

