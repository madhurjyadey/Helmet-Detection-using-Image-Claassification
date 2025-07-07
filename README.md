# Helmet-Detection-using-Image-Claassification
# 1.Introduction
By taking a brief idea from the NRL Project Safety Assistance, the project is about the detection if the person is wearing helmet or not through image classification techniques in Machine Learning and Computer Vision. In industrial environments like oil refineries, ensuring the safety of personnel, it
is the mandatory use of *Personal Protective Equipment (PPE)* such as helmets.
# 2.Objective
The primary goal of this project is to develop an image classification model capable of detecting whether a person in an image is wearing a helmet or not. This aids in enhancing safety compliance and reducing the risk of injuries in workplaces and traffic environments.
# 3.Tools and Technologies Used
## 3.1 **Programming Language**: Python
## 3.2 **Libraries**: PyTorch, YOLO
## 3.3 **Dataset**: Taken From RoboFlow (https://universe.roboflow.com/wedothings/hard-hat-detector-znysj/dataset/1)
## 3.4 **Development Platform**: Virtual Studio Code
# 4.Methodology
## 4.1. Data Collection:
The first step in developing a machine learning model for tasks like helmet detection is data collection. This involves gathering a large and diverse set of labelled images that show people both wearing and not wearing helmets.
## 4.2. Data Preprocessing:
Once the data is collected, it needs to be prepared for training, which includes resizing all images to a uniform dimension, normalizing pixel values to fall within a standard range, and applying data augmentation techniques. The dataset is then split into training, validation, and test sets to facilitate proper
evaluation during and after training.
## 4.4. Model Design:
Convolutional Neural Networks (CNNs) are commonly used due to their ability to detect spatial patterns in images. The model typically includes convolutional layers to extract features, pooling layers to reduce dimensionality, and fully connected layers to make the final classification decision.
## 4.5. Compilation:
By selecting a loss function appropriate for the task, an optimizer such as Adam or SGD to adjust the model&#39;s weights,and evaluation metrics like accuracy, precision, and recall to monitor performance.
## 4.5. Model Training:
During training, the model processes the training images in smaller groups called batches. For each batch, the model makespredictions, compares them to the actual labels, calculates the error (loss), and updates the weights using backpropagation.This process continues until all data has been seen once,marking the end of one epoch.
## 4.6. Validation:
After each epoch, the model&#39;s performance is evaluated using the validation dataset. This helps to monitor whether the model is overfitting or generalizing well to new data.
## 4.7. Testing:
Once training is complete, the final evaluation is done using the test dataset, which contains images the model has never seen before.
## 4.8. Prediction:
After successful testing, the trained model can be used to make predictions on new, real-world images. Given an input image, the model will output a probability or label indicating whether a helmet is present or not.
## 4.9. Deployment:
Finally, the model is deployed into a practical application. It can be embedded in a surveillance system, a mobile app, or a factory monitoring tool. The trained model file is loaded, and live video streams or images are fed to it in real-time to detect helmet usage and generate alerts for violations.

# 5. Model Outputs and Visualizations
![image](https://github.com/user-attachments/assets/0670e4f9-bc04-456e-8ce8-6543bb013eca)

Fig: The following code snippet is to run the train data

![image](https://github.com/user-attachments/assets/c649b758-5bf5-45a2-bbe5-9cc2fa64673b)

Fig: The following train data is run in the windows command prompt

![image](https://github.com/user-attachments/assets/f0c17850-5c81-4d9c-8ada-431615ae63c1)

Fig: The terminal starts the epoch process

![image](https://github.com/user-attachments/assets/a6506012-72b3-494e-b2de-c3ae7f592540)

Fig: The ending phase of the epoch with accuracy given to the classes present

![image](https://github.com/user-attachments/assets/93922edf-0714-4873-972d-dc87a5bac731)

Fig: The code snippet to run the test dataset

![image](https://github.com/user-attachments/assets/fc66b083-0587-4cbb-97d8-96fba93255df)

Fig: The following test data is run in the windows command prompt

![image](https://github.com/user-attachments/assets/94822029-4c9f-4cb0-bc9b-a505a083ecc4)

Fig: The following Output

# 6. Statistical Results:

![confusion_matrix](https://github.com/user-attachments/assets/7829ed16-f1d8-411b-ae2b-98993e513367)

Fig: Confusion Matrix for Helmet Detection Model

This confusion matrix visualizes the raw prediction results of the helmet detection model across all classes, including “Helmet_on_head,”
“Helmet_off_head,” “Helmet_poor_fit,” “hat,” “no_helmet,” “wrong_helmet_type,” and “background.” The diagonal cells represent correct predictions,
while off-diagonal cells indicate misclassifications. The matrix helps identifywhich classes are most frequently confused, guiding further model
improvement.

![confusion_matrix_normalized](https://github.com/user-attachments/assets/28457507-52c8-486d-935a-dc4c898a4e52)

Fig: Normalized Confusion Matrix for Helmet Detection Model

This normalized confusion matrix presents the proportion of correct andincorrect predictions for each class, with values ranging from 0 to 1. High
values along the diagonal indicate strong model performance for thoseclasses. Normalization allows for clearer comparison between classes,especially when class distributions are imbalanced.
![F1_curve](https://github.com/user-attachments/assets/88d3ae00-bc51-4726-b682-f55698ea2adb)

Fig: F1-Confidence Curve for Each Class

The F1-Confidence curve illustrates the relationship between the model’s confidence threshold and the F1-score for each class. Each line represents a
different class, while the thick blue line shows the average F1-score across all classes. This graph helps determine the optimal confidence threshold to
maximize the model’s overall precision and recall balance.

![labels](https://github.com/user-attachments/assets/ecbd4d3b-97e3-4337-b536-734ccc3cee64)

Fig: Dataset Distribution and Bounding Box Analysis

The top-left bar chart shows the distribution of labelled instances for each class in the dataset, highlighting class balance. The top-right plot overlays all
bounding boxes to visualize their spatial distribution. The bottom two scatter plots display the normalized centre coordinates (x, y) and the width/height of bounding boxes, providing insight into object size and location diversity in the dataset.

![P_curve](https://github.com/user-attachments/assets/d5f07bca-658d-4b1b-958e-ac91dc297fa1)

Fig: Precision-Confidence Curve

This graph displays the relationship between model confidence and precision for each class in the helmet detection task. Each line represents a different
class, while the thick blue line shows the average precision across all classes. The plot helps determine the optimal confidence threshold to maximize
precision, ensuring that predictions labelled as “helmet” or “no helmet” are reliable.

![PR_curve](https://github.com/user-attachments/assets/7ada66b4-a9f2-4ace-9685-f9867f4c43c5)

Fig: Precision-Recall Curve

This curve illustrates the trade-off between precision and recall for each class in the helmet detection model. The area under each curve (mAP@0.5)
is a key metric for evaluating detection performance. The thick blue line represents the average performance across all classes, with higher curves
indicating better model capability to balance precision and recall

![R_curve](https://github.com/user-attachments/assets/3ac4ee63-5522-4e2c-9fd3-9f558f45d717)

Fig: Recall-Confidence Curve

This diagram shows how recall varies with the model’s confidence threshold for each class. The thick blue line indicates the overall recall trend across all
classes. This curve is useful for selecting a threshold that maximizes the model’s ability to detect all true helmet and non-helmet cases, minimizing
missed detections.

![results](https://github.com/user-attachments/assets/9ffb4b21-4606-4ea6-a1b5-caeab4877b53)

Fig: Training and Validation Metrics Over Epochs

This composite figure tracks key training and validation metrics over 100 epochs. The top row shows training losses (box, class, and DFL losses) and
precision/recall, while the bottom row presents validation losses and mean average precision (mAP) scores. The decreasing loss values and increasing
mAP indicate effective model learning and improved detection accuracy with more training.

## 7. Results
![image](https://github.com/user-attachments/assets/817c15ac-2ad0-4d47-8d8a-fa294e6d989f)

Fig: Model Output—No Helmet Detected

The model accurately identified the absence of a helmet, labeling the individual as “no_helmet” with high confidence. This demonstrates the system’s effectiveness in flagging safety non-compliance before protective equipment is worn.

![image](https://github.com/user-attachments/assets/0f6391f2-e703-4e54-946d-8ba7f180b876)

Figure 2: Model Output—Helmet Being Worn

As the helmet is being put on, the model successfully detected and labelled the action as “Helmet_on_head,” indicating its ability to recognize the
transition to compliance with safety protocols.

# 8. Advantages of the Project

## 8.1. Real-Time Monitoring:
Machine learning models, especially those integrated with video surveillance systems, enable real-time helmet detection. This allows for immediate identification of non-compliance, improving safety response times in industrial settings.
## 8.2. High Accuracy and Consistency:
Unlike manual monitoring, which is prone to human error and fatigue, trained models provide consistent and reliable detection accuracy, even in complex or crowded environments.
## 8.3. Scalability:
Once developed, the model can be deployed across multiple locations without the need to train new staff. It can also be scaled to process inputs from several cameras simultaneously with minimal additional cost.
## 8.4. Improved Workplace and Road Safety:
Early detection and alerts help in enforcing helmet usage, which directly contributes to reducing injuries and fatalities caused by head injuries in both
industrial and road traffic scenarios.
## 8.5. Integration with Other Systems:
The model can be integrated with dashboards for alerts, reporting, or even denial of entry to those not wearing helmets. This enhances the overall safety
infrastructure.
## 8.6. Learning and Improvement Over Time:
With more data and feedback, the model can be retrained to improve performance and adapt to new environments or helmet designs, making it more robust over time.

# 9. Conclusion
The implementation of a machine learning-based helmet detection system presents a powerful solution to enhance safety compliance industrial environments. By leveraging image classification techniques, such systems can automatically and accurately identify whether individuals are wearing helmets, significantly reducing the reliance on manual supervision. The model operates with high accuracy, consistency, and speed, making it ideal for real-time applications such as surveillance monitoring and access control. As technology advances, such systems will continue to improve, offering even more robust and intelligent
