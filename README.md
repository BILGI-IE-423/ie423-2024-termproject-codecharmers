[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/N24Xct0L)

Pınar Sude Gürdamar - 120203073

Fatma Hilal Börklü - 119200087

Ece Taş - 121203066

İlayda İyikesici - 119203041

Emel Çatiçdaba - 121203072



# Abstract 

This study aims to comprehensively analyze the factors influencing mobile app ratings. Understanding these elements is essential to improving user satisfaction and optimizing app performance. The study looks into the application of machine learning algorithms and how variables like user reviews, app cost, and genre affect app ratings. It also examines the emotional reactions of users both before and after the COVID-19 outbreak. In general, the study offers significant perspectives for marketers and app developers to enhance customer contentment and maximize app efficiency.


# Scope of the Project

This project, conducted using comprehensive datasets obtained from Google Play Store and other mobile application platforms, aims to understand the market performance, user experience, and emotional responses of mobile applications. The project will extensively analyze fundamental features such as application names, categories, rating scores, download counts, and sentiment analysis results derived from user reviews.

Key focuses of the project include:

- The study leverages datasets from Google Play Store and other mobile platforms to comprehend the market performance, user experience, and emotional responses associated with mobile applications.

- It will delve into basic features like application names, categories, rating scores, download counts, along with detailed analysis of sentiment analysis results from user reviews, aiming to provide in-depth insights into how each application is perceived among users.

- Comparative analysis will evaluate changes in user perceptions of applications before and after the Covid-19 period.


# Research Questions

Which features play a more decisive role in influencing app scores: user reviews, app pricing, or app genre? Which features can be best utilized to enhance the accuracy of the score prediction model?

Which machine learning techniques may be used to fully comprehend the elements influencing the app installations' rating? How do these algorithms utilize features such as emotional content, pricing policies, user reviews, and other external factors to predict app scores effectively?

Considering the emotional changes in user evaluations before and during COVID-19, what particular machine learning and sentiment analysis approaches can be used? In what ways do these studies most accurately capture the changes in users' emotional reactions before and after the epidemic in various spheres of life?


# Related Works

The internet's growth has allowed users to share their opinions on social media and commercial websites. As a result, analyzing sentiments and emotions in text has become an important area of research. This field focuses on automatically classifying user reviews to gain insights into public sentiment.

Researchers usually classify customer reviews into three categories: positive, negative, or neutral. But, since reviews can be super positive or super negative, using a specific scale to measure how positive or negative they are could make sentiment analysis work better. (Singh et al., 2016).

One way to perform sentiment analysis is by the use of lexicons, which assign a sentiment value to each term. To enhance accuracy in calculating sentiment values from basic summation and mean methods, Jurek et al. (2015) suggested a normalization function. Lexicon-based methods can be categorized into two types: dictionary-based and corpus-based approaches.

In dictionary-based methods, a list of initial words is created and then expanded with words that have similar or opposite meanings (Schouten and Frasincar, 2015). On the other hand, corpus-based methods involve identifying sentiment words that are specific to a particular subject based on their usage in context (Bernabé-Moreno et al., 2020). Another approach suggested by Cho et al. (2014) is a three-step method to improve how polarity is determined based on context, as well as making dictionaries more adaptable for different domains.

Within the field of text classification, researchers have developed several techniques. For example, Dai et al. (2007) used an iterative Expectation-Maximization algorithm to transfer a Naïve Bayes classifier from one domain to another. This method allowed them to apply the classifier to a new context effectively. In their 2008 study, Gao et al. employed a combination of multiple classifiers trained on various source domains to classify target documents by assessing their similarity to a clustering of the target documents.

In cross-domain classification or domain adaptation, text categorization is essential. It was Pan and Yang (2010) who suggested this concept where knowledge can be transferred between two domains that have different distributions but the same labels.

Techniques like Latent Dirichlet Allocation (Blei et al., 2003) or Latent Semantic Indexing (Weigend et al., 1999) uncover hidden correlations among words, thereby enhancing document representations. More recent approaches extract semantic information from terms by utilizing external knowledge bases such as WordNet (Scott and Matwin, 1998) or Wikipedia (Gabrilovich and Markovitch, 2007).

Another approach to sensitivity analysis involves machine learning algorithms, where data sets are classified into training and test sets for model training and analysis. Supervised classification algorithms such as Naïve Bayes, Support Vector Machine (SVM), and decision trees are often used (Gamon, 2004). Bučar et al. (2018) developed a sentiment lexicon and labeled news corpora to analyze sentiments in Slovene texts. They found that Naïve Bayes performed better than SVM. Tiwari et al. (2020) used SVM, Naïve Bayes, and maximum entropy algorithms with n-gram feature extraction on a dataset of movie reviews. They observed that accuracy decreased as the n-gram values increased.

In different research papers, ensemble methods have been explored to tackle the hurdles of sentiment analysis by utilizing mathematical and statistical approaches like Gaussian distributions. However, these models are frequently seen as theoretical and lack real-world application (Buche et al., 2013). On the other hand, in a thorough exploration of machine learning, a separate study employed a variety of methods such as decision trees and neural networks to forecast app rankings by considering numerous features of the apps (Suleman et al., 2019). 

Ratings are crucial because they directly impact an app's visibility and success. Apps with higher ratings are­ more likely to show up in the Google­ Play Store and attract new people­ to try the app. Sentime­nt analysis explores an intriguing realm: de­coding the nuanced expre­ssions embedded within re­views, including the intricate subte­xt conveyed through emojis. Emojis can help share­ feelings that words alone might not show. Studie­s show that emojis can share how people­ feel, and can help pre­dict ratings they might give to an app. Analyzing these alongside textual reviews offers a richer, more dimensional understanding of user opinions (Martens and Johann, 2017).  

# Datasets Used


1. Google Play Store Apps
   
https://www.kaggle.com/datasets/lava18/google-play-store-apps?select=googleplaystore.csv

2. Google Play Store Apps - User Reviews

https://www.kaggle.com/datasets/lava18/google-play-store-apps?select=googleplaystore_user_reviews.csv

3. Google Play Store Apps Reviews (+110K Comment)-Apps

https://www.kaggle.com/datasets/mehdislim01/google-play-store-apps-reviews-110k-comment?select=Apps.csv

4. Google Play Store Apps Reviews (+110K Comment)-Reviews

https://www.kaggle.com/datasets/mehdislim01/google-play-store-apps-reviews-110k-comment?select=Reviews.csv



# About Preprocessing Data

**Loading Data**

The initial step in the project involves loading the data from the provided CSV files. Utilized four main datasets, each containing specific information about Google Play Store applications and their user reviews. Below are the details of the files used and their contents:

1.	googleplaystore.csv:
o	This file contains comprehensive information about various applications available on the Google Play Store. The data includes details such as app names, categories, ratings, number of reviews, size, installs, type, price, content rating, genres, last updated date, current version, and Android version required.
2.	googleplaystore_user_reviews.csv:
o	This file includes user reviews for the applications listed in the googleplaystore.csv. The data includes the app name, the review text, the sentiment (positive, negative, or neutral), and the sentiment polarity and subjectivity scores.
3.	Apps.csv:
o	Similar to googleplaystore.csv, this file contains information about various applications, from the Covid-19 period. It includes similar attributes such as app names, categories, ratings, number of reviews, size, installs, type, price, content rating, genres, last updated date, current version, and Android version required.
4.	Reviews.csv:
o	This file includes user reviews for the applications listed in Apps.csv. It contains details such as the app name, the review text, the translated review text, the sentiment (positive, negative, or neutral), and the sentiment polarity and subjectivity scores.

After loading the data, merged the datasets to create a comprehensive DataFrame containing all relevant information about the apps and their reviews. The merging process involves creating three main DataFrames: mergedMain, mergedChild, and mergedAll.

1. mergedMain
Combines app data from dfParentApps with their respective user reviews from dfParentReviews.
2. mergedChild
Combines app data from dfChildApps with their respective user reviews from dfChildReviews.
3. mergedAll
Combines the previously merged DataFrame mergedMain with selected columns from mergedChild to include additional review information.


**Handling Outliers**

Managing the outliers the process is straightforward. For every column, the first quartile (Q1) and the third quartile (Q3) which represent the 25th and 75th percentiles were computed. Subsequently, the Interquartile Range (IQR) as the difference between Q3 and Q1, for a better understanding of the spread of the middle 50% of the data. The acceptable range is then determined by computing the upper and lower bounds. Which are set to be '±1.5 x IQR'. Any values outside this range are considered outliers and are replaced with NaN to prevent them from skewing the analysis. 

**Data Cleaning and Formatting**

To prepare the dataset for analysis, several cleaning and formatting steps are performed. First, the 'Size' column was standardized by replacing 'Varies with device' entries with NaN, removing 'M' suffixes for megabytes and converting 'k' suffixes for kilobytes to scientific notation (e-3), ensuring all values were numeric. Rows with missing 'Review Date' entries were dropped to maintain data integrity. Additionally, rows with NaN or non-numeric values in the 'Size' column were also removed to ensure consistency. Numeric columns like 'Installs' were cleaned by removing special characters and converting them to integers, while 'Price' underwent transformations to remove dollar signs and convert to floats. 'Size' and 'Rating' columns were explicitly converted to float types for uniformity. 'Review Date' was converted to datetime format and reformatted to 'dd-mm-yyyy' to standardize its presentation. Finally, missing values in numeric columns were filled with zeros to facilitate accurate analysis. 


**Scaling Numeric Features and Encoding Categorical Variables**

Numeric features such as 'Rating', 'Reviews', 'Size', 'Installs', 'Sentiment_Polarity', and 'Sentiment_Subjectivity' are standardized using StandardScaler() from scikit-learn. Ensuring that features with varying scales contribute equally without dominance due to the higher magnitude, is achieved by standartizing the range of numeric data through scaling. This transformation centers the data around zero and scales it to have unit variance.

Categorical variables including 'Category', 'Type', 'Content Rating', and 'Genres' are transformed into numerical labels using LabelEncoder(). Each unique category within these columns is assigned a unique integer, allowing categorical data to be effectively utilized in machine learning algorithms that require numeric inputs.

The scaled numeric features and encoded categorical variables are merged along the columns axis. The resulting DataFrame "final_df" integrates both types of transformed data, ensuring that all features are now in a format suitable for machine learning tasks.

Utilizing this combined dataset "final_df", analysis and predictions may now be performed without any need for additional preprocessing.

Splitting the Data for Training and Testing

The final dataset is first splitted into input features (x) and target value (y). The input features (X) encompass all columns in final_df except for 'Rating', which serves as our target variable.

The parameter "test_size=0.2" specifies that 20% of the data will be reserved for testing, while the remaining 80% will be allocated for training the machine learning models. The "random_state=42" parameter ensures reproducibility by fixing the random seed, thereby ensuring consistent results across different executions.

The training set (X_train and y_train) is used to capture the patterns and the relationships within the data. Models learn from pairing the input features (X_train) and known target values (y_train).

The test set (X_test and y_test) serves as an independent dataset used to evaluate the trained models' performance. It simulates real-world scenarios where models encounter new, blind, previously unseen data.

**Exploratory Data Analysis (EDA)**

For understanding the basic characteristics of the dataset and gain initial insights, basic statistical analysis is performed. 

The "describe()" function generates statistics such as count, mean, standard deviation, minimum, quartiles, and maximum for numerical columns in the dataset. With the information gathered us can understand the data range, evaluate of the quality of the data and detect possible outliers. The underlying pattern, overall distribution, central tendency of the data is overviewed.

Basic Statistics

```plaintext
            Rating         Size      Installs   Price  Sentiment_Polarity
count  6534.000000  6534.000000  6.534000e+03  6534.0         2896.000000
mean      4.453336    34.826752  1.048096e+08     0.0            0.308423
std       0.188545    21.970818  2.266175e+08     0.0            0.282917
min       4.100000     4.400000  5.000000e+05     0.0           -0.250000
25%       4.400000    15.000000  1.000000e+07     0.0            0.000000
50%       4.400000    37.000000  5.000000e+07     0.0            0.308333
75%       4.500000    53.000000  1.000000e+08     0.0            0.500000
max       4.900000    77.000000  1.000000e+09     0.0            1.000000

       Sentiment_Subjectivity
count             2896.000000
mean                 0.415136
std                  0.279582
min                  0.000000
25%                  0.000000
50%                  0.502976
75%                  0.600000
max                  0.900000
```


![download](https://github.com/BILGI-IE-423/ie423-2024-termproject-codecharmers/assets/167031646/f20c4202-fc48-4331-b91e-1bd9435e4ed7)
*Figure 1.1: Correlation Between Numerical Features*

![download](https://github.com/BILGI-IE-423/ie423-2024-termproject-codecharmers/assets/167031646/c3b5f5b0-25fb-4ace-a07d-b4cbed17e740)
*Figure 1.2: Pair Plot of Numeric Features*

![download](https://github.com/BILGI-IE-423/ie423-2024-termproject-codecharmers/assets/159184426/a05682c7-e983-41c1-ae4a-b76e12206d2e)



# The score prediction model and Features


When machine learning models are evaluated, especially when the results obtained from Random Forest , Linear Regression and Gradient Boosting Regressor models are examined, it is clearly seen that factors such as user comments (Reviews) and application type (Genres) affect application scores. It has been determined that especially in the Random Forest Regressor model, the "Reviews" and "Genres" features are decisive and these features are highly effective in score prediction. Therefore, user reviews and app type are among the most effective features for predicting app scores. These features can be optimally used to improve the accuracy of the score prediction model.  
User reviews are valuable because they provide direct feedback about the quality of the app and user satisfaction. Additionally, the app type also plays an important role in score estimation as it reflects the app's overall category and target audience. App pricing may also be effective in score estimation, but the effect of this factor was seen to be less pronounced compared to others in the models used in this study.  
While the Linear Regression model performed reasonably well with a mean square error of 0.0226 and an R-squared value of 0.5023 on the test set, the Gradient Boosting Regressor model achieved very low error rates and a high R-squared value of 0.9991, predicting application scores extremely accurately. It was seen that he did. Additionally, it was remarkable that the Random Forest Regressor model achieved accuracy scores of up to 100% in the training and test sets, which tends to overfit the training data, showing that especially the "Reviews" (75.654%) and "Genres" (24.346%) features were decisive in score prediction. On correlation matrix below, it is clearly seen that there is strongest correlation is the positive correlation between "Genre" and "Reviews". These results show that the most effective features for predicting app ratings are user reviews and app type.

![Ekran görüntüsü 2024-06-17 154033](https://github.com/BILGI-IE-423/ie423-2024-termproject-codecharmers/assets/159184426/6c406f68-84d7-46d7-a45d-f9c408b99602)  
*Figure 2.1: Actual vs Predicted Ratings (Linear Regression)*
![Ekran görüntüsü 2024-06-17 154119](https://github.com/BILGI-IE-423/ie423-2024-termproject-codecharmers/assets/159184426/8210408c-89d4-4f34-bfcd-e8c33c6d7d5b)  
*Figure 2.2: Actual vs Predicted Ratings (Gradient Boosting Regressor)*
![WhatsApp Görsel 2024-06-18 saat 00 57 48_42400179](https://github.com/BILGI-IE-423/ie423-2024-termproject-codecharmers/assets/159184426/05fb704d-2e9a-4259-8d32-6097266ef517)
*Figure 2.3: Actual vs Predicted Ratings (Random Forest)*
![download](https://github.com/BILGI-IE-423/ie423-2024-termproject-codecharmers/assets/159184426/f2490c71-90d4-433e-81f7-a0c38a2e714f)
*Figure 2.5: Correlation Matrix for Features*
![Ekran görüntüsü 2024-06-17 154326](https://github.com/BILGI-IE-423/ie423-2024-termproject-codecharmers/assets/159184426/7b79ca6f-91f5-4068-9d35-e6d51d37cca6)  
*Figure 2.6: Feature Importance for App Ratings*





# Emotional Studies: Before and After the Pandemic

The studies presented in the graphs on “SentimentAnalysisGraphs.ipynb” capture the changes in users' emotional reactions before and after the COVID-19 pandemic in several significant ways.By reading the datasets, column names were standardized and datasets were merged. Sentiment analysis was performed on the “content” column using TextBlob from the merged dataset. This analysis resulted in the calculation of sensitivity, polarity, and subjectivity values ​​for each review. The data was divided into pre-COVID-19 and post-COVID-19 periods and saved in separate CSV files. Both CSV files included sentiment analysis results based on the "content" column. This process aimed to analyze sentiment in app reviews and their changes over time. Here's everything in detail:

**Pre-COVID Sentiment Distribution**: The pie chart indicates that before the pandemic, a majority of the sentiments were positive (65.1%), with negative sentiments at 26.1%, and neutral sentiments at 8.9%.  
**Post-COVID Sentiment Distribution**: The second pie chart shows a slight decrease in positive sentiments to 61%, a small decrease in negative sentiments to 24.4%, and an increase in neutral sentiments to 14.6%.

![Ekran görüntüsü 2024-06-17 124639](https://github.com/BILGI-IE-423/ie423-2024-termproject-codecharmers/assets/159184426/0b46f365-c378-4ef4-958b-a819b8dd258f)

*Figure 2.7: Sentiment Distribution Pre-COVID-19*
![Ekran görüntüsü 2024-06-17 124647](https://github.com/BILGI-IE-423/ie423-2024-termproject-codecharmers/assets/159184426/34a469e9-4f1a-4e2f-9be4-2ef9a2fcfbf6)

*Figure 2.8: Sentiment Distribution Post-COVID-19*

The emotion distribution bar chart comparing pre-,and post-COVID reveals an obvious decrease in positive emotions and a little decline in negative emotions, with a notable increase in neutral sentiments following the COVID-19 pandemic. This implies that consumers' emotional responses during the epidemic were less judgmental or unsure.

![Ekran görüntüsü 2024-06-17 124615](https://github.com/BILGI-IE-423/ie423-2024-termproject-codecharmers/assets/159184426/a7438008-0c94-4303-b2d1-d1f638f19cd4)  
*Figure 2.9: Sentiment Distribution Pre- and Post-COVID-19*

The line graph shows the evolution of sentiment trends over time; it shows an obvious decrease in positive emotions prior to COVID-19 and an increase in neutral feelings in response, while negative sentiments stay mostly unchanged. This pattern shows a change in user responses, which may have been impacted by the pandemic's stress and uncertainty on a worldwide scale.

![Ekran görüntüsü 2024-06-17 124703](https://github.com/BILGI-IE-423/ie423-2024-termproject-codecharmers/assets/159184426/d57afbe2-74bc-4a23-9da3-e15b7a9c401a)  
*Figure 2.10: Sentiment Trends in Pre- and Post-COVID-19*

**Pre-COVID Genre Usage**: The bar chart shows high usage of Education and Photography apps before the pandemic, with other genres having significantly lower usage rates.  
**During COVID Genre Usage**: There's a noticeable shift in genre usage during the pandemic, with Education and Photography remaining dominant, but there's a slight decrease in their proportion, indicating a diversification in app usage.

![Ekran görüntüsü 2024-06-17 132444](https://github.com/BILGI-IE-423/ie423-2024-termproject-codecharmers/assets/159184426/99834023-aa77-4f5d-8633-5adea258bc9e)  
*Figure 2.11: Pre-COVID Genre Usage*
![Ekran görüntüsü 2024-06-17 124727](https://github.com/BILGI-IE-423/ie423-2024-termproject-codecharmers/assets/159184426/461a2d21-c1b9-4359-af50-c386203f9849)  
*Figure 2.12: During COVID Genre Usage*

The comparison bar chart illustrates how consumption trends for different app genres changed. For instance, during the pandemic, there was a noticeable rise in the use of Tools and Art-Design apps, indicating changes in user requirements and activities. In contrast, Education and Photography apps continued to have significant utilization

![Ekran görüntüsü 2024-06-17 124739](https://github.com/BILGI-IE-423/ie423-2024-termproject-codecharmers/assets/159184426/2e554eb2-1dd1-4698-9950-949943f1c1ff)  
*Figure 2.13: Genre Usage Comparision*

Before and during the epidemic, the stacked bar chart offers a thorough analysis of the sentiment distribution across several app categories. Although there is an obvious increase in neutral opinions across all categories throughout the pandemic, areas with significant positive sentiments include Tools, Health & Fitness, and Education. This shows the need to apps that are in Tools, Health & Fitness, and Education genres increased during pandemic. Also, these apps made lives easier and changed the dynamics of education, workout, and business life on daily basis. 

![Ekran görüntüsü 2024-06-17 124801](https://github.com/BILGI-IE-423/ie423-2024-termproject-codecharmers/assets/159184426/2fc3fbaf-74d4-498d-9994-294be79a2ccf)  
*Figure 2.14: Sentiment Distribution by Category*  

# Future Works

To improve the project that includes score prediction models and insights, the following steps can be implemented to make the scope of the project more comprehensive:

**Data Augmentation:**
Data augmentation artificially increases the training set by creating modified copies of a dataset using existing data. Generating synthetic data to augment the training set, especially for underrepresented categories or features, might make our model more reliable and stronger.

**Stacking:**
Combining predictions from multiple models (e.g., Random Forest, Gradient Boosting, Linear Regression) would create a more robust and accurate ensemble model.

**Topic Modeling:**
Latent Dirichlet Allocation (LDA) is a probabilistic model that generates a set of topics, each represented by a distribution over words, for a given corpus of documents. Implementing LDA to identify key themes and topics within user reviews would significantly improve our predictions.

**Deep Learning Models:**
To improve the model, implementing learning architectures such as LSTM and Transformer models might be effective for handling sequential and text data.

**Emotion Detection:**
Using emotion detection algorithms to classify reviews into specific emotional categories (e.g., joy, anger, sadness) would enhance the project’s comprehensiveness in emotion studies.

**User Segmentation:**
Applying clustering algorithms (e.g., K-means) to segment users based on their reviews, ratings, and installation behaviors is preferable to gain deeper insights into user behavior and preferences through segment specific analytics. This would allow the development of personalized recommendations and targeted marketing strategies based on user segments.



# Project Timeline and Deliverables

```mermaid
gantt
    title Machine Learning Project Timeline

    dateFormat  YYYY-MM-DD
    section Problem Analysis
    Importance and Impact of the Problem        :done,    p1, 2024-03-01, 2024-03-15
    Nature of the Problem                       :done,    p2, 2024-03-01, 2024-03-15
    Determination of Success Criteria           :done,    p3, 2024-03-01, 2024-03-15
    Formulation of Research Questions           :done,    p4, 2024-03-01, 2024-03-15

    section Data Collection
    Identification of Potential Data Sources    :done,    d1, 2024-03-16, 2024-03-30
    Evaluation of Data Accuracy and Cleanliness :done,    d2, 2024-03-16, 2024-03-30
    Consideration of Ethical and Legal Constraints :done, d3, 2024-03-16, 2024-03-30
    Proposal Writing                            :done,    d4, 2024-03-16, 2024-03-30

    section Data Preprocessing
    Handling Missing and Outlier Values         :done,     dp1, 2024-04-01, 2024-04-15
    Data Transformation and Normalization       :done,     dp2, 2024-04-01, 2024-04-15
    Feature Engineering and Encoding Techniques :active,   dp3, 2024-04-01, 2024-04-15

    section Exploratory Data Analysis (EDA)
    Sentiment Analysis from Reviews             :         eda1, 2024-04-16, 2024-05-01
    Impact Assessment of the COVID-19 period    :         eda2, 2024-04-16, 2024-05-01

    section Model Building (Splitting Train and Test Sets)
    Selection of Algorithms and Splitting      :         mb1, 2024-05-02, 2024-05-15
    Model Training and Evaluation              :         mb2, 2024-05-02, 2024-05-15
    Hyperparameter Tuning and Adjustment       :         mb3, 2024-05-02, 2024-05-15

    section Model Evaluation
    Selection and Application of Metrics       :         me1, 2024-05-16, 2024-05-30
    Assessment of Confusion Matrix, ROC Curve  :         me2, 2024-05-16, 2024-05-30
    Evaluation of Model Generalization         :         me3, 2024-05-16, 2024-05-30

    section Model Optimization
    Methods for Hyperparameter Optimization    :         mo1, 2024-06-01, 2024-06-15
    Techniques for Feature Selection           :         mo2, 2024-06-01, 2024-06-15
    Ensemble Methods and Stacking              :         mo3, 2024-06-01, 2024-06-15

    section Model Deployment
    Transitioning the Model into Production   :         md1, 2024-06-16, 2024-06-19
    Accessibility Through APIs and Services   :         md2, 2024-06-16, 2024-06-19
    Performance and Scalability Optimization :         md3, 2024-06-16, 2024-06-19

   ```


## REFERENCES 

A. Buche, D. Chandak, and A. Zadgaonkar, Opinion mining and analysis: a survey, arXiv preprintarXiv:1307.3336, 2013

Bernabé-Moreno J, Tejeda-Lorente A, Herce-Zelaya J, Porcel C, Herrera-Viedma E (2020) A context-aware embeddings supported method to extract a fuzzy sentiment polarity dictionary. Knowledge-Based Systems 190:105236.

Blei, D. M., Ng, A. Y., and Jordan, M. I. (2003). Latent Dirichlet allocation. The Journal of Machine Learning research, 3:993–1022.

Bučar, Jože & Žnidaršič, Martin & Povh, Janez. (2018). Annotated news corpora and a lexicon for sentiment analysis in Slovene. Language Resources and Evaluation. 52. 10.1007/s10579-018-9413-3.

Cho H, Kim S, Lee J, Lee JS (2014) Data-driven integration of multiple sentiment dictionaries for lexicon-based sentiment classification of product reviews. Knowledge-Based Systems 71:61–71.

C. Shin, J.-H. Hong, and A. K. Dey, ‘‘Understanding and prediction of
mobile application usage for smart phones,’’ in Proc. ACM Conf. Ubiquitous Comput. (UbiComp), 2012, pp. 173–182.

D. Martens and T. Johann, On the emotion of users in appreviews, in Proc. IEEE/ACM Int. Workshop Emotion Awareness Softw. Eng. (Buenos Aires, Argentina), May 2017

Dai, W., Xue, G.-R., Yang, Q., and Yu, Y. (2007). Transferring naive bayes classifiers for text classification. In Proceedings of the AAAI ’07, 22nd national conference on Artificial intelligence, pages 540–545.

Gabrilovich, E. and Markovitch, S. (2007). Computing semantic relatedness using Wikipedia-based explicit semantic analysis. In Proceedings of the 20th International Joint Conference on Artificial Intelligence, volume 7, pages 1606–1611.

Gamon M (2004) Sentiment classification on customer feedback data: noisy data, large feature vectors, and the role of linguistic analysis. In: COLING 2004: Proceedings of the 20th international conference on computational linguistics, pp 841–847

Gao, Yan & Mas, Jean. (2008). A comparison of the performance of pixel based and object based classifications over images with various spatial resolutions. Online Journal of Earth Science. 2. 27-35.

Jurek-Loughrey, Anna & Mulvenna, Maurice & Bi, Yaxin. (2015). Improved lexicon-based sentiment analysis for social media analytics. Security Informatics. 4. 10.1186/s13388-015-0024-x.

 K. Huang, C. Zhang, X. Ma, and G. Chen, ‘‘Predicting mobile application usage using contextual information,’’ in Proc. ACM Conf. UbiquitousComput. (UbiComp), 2012, pp. 1059–1065.

 M. Suleman, A. Malik, and S. S.Hussain,Google play storeapp ranking prediction using machine learning algorithm, Urdu News Headline, Text Classification by Using Different Machine Learning Algorithms, 2019.

Pan, S. J., Kwok, J. T., and Yang, Q. (2008). Transfer learning via dimensionality reduction. In Proceedings of the AAAI ’08, 23rd national conference on Artificial intelligence, pages 677–682.


S. J. Pan and Q. Yang, "A Survey on Transfer Learning," in IEEE Transactions on Knowledge and Data Engineering, vol. 22, no. 10, pp. 1345-1359, Oct. 2010, doi: 10.1109/TKDE.2009.191.

Sam Scott and Stan Matwin. 1998. Text Classification Using WordNet Hypernyms. In Usage of WordNet in Natural Language Processing Systems.

Schouten, Kim & Frasincar, Flavius. (2015). Survey on Aspect-Level Sentiment Analysis. IEEE Transactions on Knowledge and Data Engineering. 28. 1-1. 10.1109/TKDE.2015.2485209.


Singh, Mangal & Nafis, Md Tabrez & Mani, Neel. (2016). Sentiment Analysis and Similarity Evaluation for Heterogeneous-Domain Product Reviews. International Journal of Computer Applications. 144. 16-19. 10.5120/ijca2016910112.

Tiwari P, Mishra BK, Kumar S, Kumar V (2020) Implementation of n-gram methodology for rotten tomatoes review dataset sentiment analysis. In: Cognitive analytics: concepts, methodologies, tools, and applications, IGI Global, pp 689–701.

X. Zou, W. Zhang, S. Li, and G. Pan, ‘‘Prophet: What app you wish touse next,’’ in Proc. ACM Conf. Pervasive Ubiquitous Comput. AdjunctPublication, 2013, pp. 167–170.

Weigend, A. S., Wiener, E. D., and Pedersen, J. O. (1999). Exploiting hierarchy in text categorization. Information Retrieval, 1(3):193–216.

