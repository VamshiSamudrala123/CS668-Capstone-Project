**Deep Learning for Lung and Colon Cancer Classification**


Vamshi Samudrala


**Project Overview**  
This project focuses on leveraging deep learning models to improve the accuracy and efficiency of lung and colon cancer diagnosis from histopathological images. By combining state-of-the-art techniques and a robust dataset, it aims to support pathologists in making faster and more reliable diagnoses.  

 **Research Questions**  
1. **Can deep learning models classify lung and colon histopathological images into specific categories (e.g., colon adenocarcinoma, benign colonic tissue, lung adenocarcinoma, lung squamous cell carcinoma, benign lung tissue) with high accuracy?**  
2. **How can these models be integrated into clinical workflows to assist pathologists effectively?**  

**Dataset: LC25000**
- **Origin**: Created by researchers at the University of California, Irvine.  
- **Description**: A large-scale dataset containing 25,000 histopathological images of lung and colon tissues, equally distributed across five categories: colon adenocarcinoma, benign colonic tissue, lung adenocarcinoma, lung squamous cell carcinoma, and benign lung tissue.  
- **Image Resolution**: All images are 768x768 pixels in JPEG format.  
- **Source**: [LC25000 Lung and Colon Histopathological Dataset - Academic Torrents](https://academictorrents.com/details/5d7d4b76380c47168027f6e2d58aa6d2fefb08b1).  

**Methodology**
1. **Data Preparation**:  
   - Converted all images to grayscale for consistent preprocessing.  
   - Applied data augmentation techniques to increase variability and avoid overfitting.  

2. **Modeling**:
   - Implemented and compared **ResNet-50** and a **custom CNN model** for classification.  
   - Fine-tuned ResNet-50 and applied **class-specific image processing (CSIP)** for improved accuracy.  
   - Models were evaluated using metrics such as accuracy, precision, recall, and F1-score.  

3. **Optimization and Evaluation**:  
   - ResNet-50 achieved a **99% accuracy**, while the custom CNN model achieved **97% accuracy** after fine-tuning and CSIP.  

**Results**  
- **ResNet-50**: High performance with better generalization and clinical applicability.  
- **Custom CNN Model**: Performed well but required additional tuning for comparable results.  


**Motivation**
- **Technical Motivation**: To advance medical image analysis using deep learning techniques for real-world healthcare applications.  
- **Personal Motivation**: To contribute to healthcare innovation and further specialize in AI-driven solutions for medical challenges.  
