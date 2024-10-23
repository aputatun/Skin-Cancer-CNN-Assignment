# Melanoma Detection Using a Custom CNN
> This project is about building a custom Convolutional Neural Network (CNN) in TensorFlow to detect melanoma, which is a type of dangerous skin cancer. By detecting melanoma early, doctors can improve survival rates, so the goal is to create a reliable model that can assist in diagnosis.

## Table of Contents
* [General Information](#general-information)
* [Technologies Used](#technologies-used)
* [Conclusions](#conclusions)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)

## General Information
- **Project Background**: To build a CNN based model which can accurately detect melanoma. Melanoma is a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. A solution that can evaluate images and alert dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis.
- **Dataset Information**: 
  - The dataset consists of 2,357 images from the **International Skin Imaging Collaboration (ISIC)**, a well-known source for skin cancer data.
  - There are nine different categories of skin conditions in the dataset:
    1. Actinic keratosis
    2. Basal cell carcinoma
    3. Dermatofibroma
    4. Melanoma
    5. Nevus
    6. Pigmented benign keratosis
    7. Seborrheic keratosis
    8. Squamous cell carcinoma
    9. Vascular lesion
  - **Class Distribution**:
    - The class with the least number of samples is **Seborrheic keratosis**, with only 77 images.
    - The dataset is dominated by classes such as:
      - **Pigmented benign keratosis**: 462 samples
      - **Melanoma**: 438 samples
      - **Basal cell carcinoma**: 376 samples
      - **Nevus**: 357 samples

## Project Pipeline
1. **Data Reading/Data Understanding**: Load the dataset and get familiar with what’s inside. Check the image types, sizes, and categories. Define where to find the training and test images.
2. **Dataset Creation**: Prepare the data for training. Divide it into training and validation sets with a batch size of 32, resizing each image to 180x180 pixels to maintain consistency.
3. **Dataset Visualization**: Create visualizations that show an example image from each category to get an understanding of the data diversity.
4. **Model Building & Training**:
   - Build a custom CNN from scratch, normalizing the pixel values to a range of 0 to 1.
   - Choose a suitable optimizer (like `Adam`) and a loss function (`categorical crossentropy`) to train the model.
   - Adam stands for Adaptive Moment Estimation. It computes adaptive learning rates for each parameter by estimating the first and second moments of the gradients.
   - Categorical Crossentropy is a commonly used loss function for classification tasks, particularly in Convolutional Neural Networks (CNNs). It measures the performance of a classification model whose output is a probability value between 0 and 1.
   - Train the model for 20 epochs and evaluate its performance, checking for any signs of overfitting or underfitting.
   - This will be our baseline experiment.
5. **Data Augmentation**: Apply data augmentation techniques (such as rotations, flips, and zooms) to the images to help the model generalize better and avoid overfitting.
6. **Training on Augmented Data**: Train the CNN again using the augmented dataset for another 20 epochs and see if this improves the results.
7. **Class Distribution Analysis**: Look at how the data is distributed among the different categories. See if any classes are underrepresented.
8. **Handling Class Imbalance**: Use the Augmentor library to add more images to the underrepresented categories, making the dataset more balanced.
9. **Training on Balanced Data**: Train the model one more time using the newly balanced dataset for 30 epochs and check if this improves the detection of rare conditions.

## Conclusions
- **Improved Model Accuracy**: Training on the augmented data helped the model perform better in identifying different skin conditions.
- **Dealing with Overfitting**: Data augmentation significantly reduced overfitting, leading to a more reliable model that performs well on unseen data.
- **Balanced Dataset Performance**: Correcting class imbalances resulted in better detection of underrepresented conditions, making the model more accurate for rarer cases.
- **Key Takeaway**: Building a custom CNN from scratch gives better control over the model’s behavior, and understanding how to handle imbalanced datasets is crucial for accurate predictions in medical data.

## Technologies Used
- **Python** - Version 3.10: The programming language used to write and execute the code.
- **TensorFlow** - Version 2.17.0: A machine learning framework that was used to create and train the custom CNN.
- **Augmentor** - Version 0.2.12: A Python library that helped in augmenting and balancing the dataset.
- **Google Colab**: A free online platform with GPU support, which made it easier to train the model faster.

## Acknowledgements
- **Data Source**: This project uses data from the [International Skin Imaging Collaboration (ISIC)](https://drive.google.com/drive/folders/1rfdJBsUWsjcBpojRrnfUjZJNtw5HzYVo?usp=drive_link)

## Contact
Created by [@Ankan Putatunda] - If you have questions or suggestions, feel free to reach out!