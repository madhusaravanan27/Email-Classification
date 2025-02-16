# **Spam Email Detection in Enron Corpus** ✉️📩  

## **Overview**  
This project focuses on **classifying emails** from the **Enron email corpus** into **spam and ham (legitimate emails)** using **machine learning techniques**. The dataset contains **3,672 ham emails** and **1,500 spam emails** that were introduced to create a balanced dataset for training classifiers.  

By leveraging **Natural Language Processing (NLP)** and **feature engineering**, we built models that effectively detect spam emails with high accuracy, ensuring a **secure and reliable email filtering system**.  


## **Data Processing**  
To prepare the data for model training, we performed the following preprocessing steps:  

- ✅ **Text Cleaning:** Removed **special characters, numbers, and URLs** to eliminate noise.  
- ✅ **Tokenization:** Used **NLTK’s word_tokenize** function to break email texts into individual words.  
- ✅ **Stopword Removal:** Filtered out common stopwords to retain meaningful words.  
- ✅ **Stemming:** Applied **Porter Stemmer** to reduce words to their root forms.  
- ✅ **Dataset Splitting:** Divided into **80% training** and **20% testing** to evaluate model performance.  

---

## **Feature Engineering**  
Feature extraction played a key role in enhancing model accuracy. The following features were engineered:  

- 📌 **Bag of Words (BoW)** – Simple word presence/absence representation.  
- 📌 **TF-IDF (Term Frequency-Inverse Document Frequency)** – Weighed word importance across emails.  
- 📌 **Emoticon Features** – Counted occurrences of emoticons commonly found in spam.  
- 📌 **Negation Handling** – Identified negated phrases for better text understanding.  
- 📌 **POS (Part-of-Speech) Tagging** – Extracted grammatical structures for spam detection.  
- 📌 **Readability Features** – Analyzed sentence complexity and length.  
- 📌 **Internet Features** – Identified repeated characters, all-caps words, and spam-specific keywords.  
- 📌 **Sentiment Scores** – Analyzed the tone of emails (positive, negative, neutral).  
- 📌 **Spam Lexicon Features** – Measured frequency of spam-associated words (e.g., *win, offer, free*).  

---

## **Classification Models & Results**  
We trained and compared multiple classifiers to identify the best-performing spam detection model:  

| **Model**                  | **F1-Score** |
|----------------------------|-------------|
| **Logistic Regression**     | **0.9597**  |
| **Random Forest**           | **0.9605**  |
| **Support Vector Machine**  | **0.9592**  |
| **Gradient Boosting**       | **0.9321**  |

📊 **Best Model:** *Random Forest* with an F1-score of **0.9605**, proving to be the most effective at distinguishing spam from ham emails.  

---

## **Model Evaluation Metrics**  
To assess the performance of our classifiers, we used the following evaluation metrics:  

- **Precision:** Accuracy of spam predictions.  
- **Recall:** Ability to detect actual spam emails.  
- **F1-Score:** Balances precision and recall for overall effectiveness.  

### **Confusion Matrix**  
- ✅ **Correctly classified spam emails:** 308  
- ✅ **Correctly classified ham emails:** 705  
- ❌ **Misclassified spam emails as ham:** 7  
- ❌ **Misclassified ham emails as spam:** 15  

These results indicate a **highly efficient spam detection model** with minimal misclassifications.  

---

## **Visualizations**  
We analyzed the dataset and model results using various **data visualizations**:  

- 📌 **Word Clouds** – Highlighted frequently occurring words in spam vs. ham emails.  
- 📌 **Bar Charts** – Displayed top words used in spam and ham emails.  
- 📌 **Confusion Matrix** – Showcased model performance in classifying emails.  
- 📌 **F1 Score Comparisons** – Compared different classifiers’ performance.  

---

## **Future Enhancements** 🚀  
We aim to improve our spam detection model by implementing:  

- 🔹 **Hybrid models** combining **deep learning (LSTMs, transformers)** and traditional ML.  
- 🔹 **Real-time email filtering** by integrating **streaming data processing (Apache Kafka, AWS Kinesis)**.  
- 🔹 **Expanded feature engineering** with **word embeddings (Word2Vec, BERT)** for enhanced spam detection.  
- 🔹 **Cross-platform integration** to extend usability across **email clients, web apps, and mobile applications**.  

---

## **Technology Stack**  
- **Machine Learning:** Logistic Regression, Random Forest, SVM, Gradient Boosting  
- **NLP Techniques:** TF-IDF, POS Tagging, Sentiment Analysis, Negation Handling  
- **Programming:** Python, NLTK, Scikit-learn, Pandas, NumPy  
- **Visualization:** Matplotlib, Seaborn, WordCloud  

---

## **Conclusion**  
This project successfully demonstrates **machine learning techniques** for **spam email detection**. By utilizing **feature engineering** and **classifier comparison**, we developed an **effective and interpretable** spam filtering model. The findings highlight **important linguistic differences** between spam and ham emails, helping advance **email security and filtering systems**.  

📌 **Next Steps:** Implement deep learning techniques to further **optimize accuracy and real-time filtering!** 🚀  

---

This README provides a **structured and concise summary** of the **Spam Email Detection** project. 🎯📧 Let me know if you need any refinements! 😊🚀  
