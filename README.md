[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/N24Xct0L)

Emel Çatiçdaba - 121203072

Ece Taş - 121203066

İlayda İyikesici - 119203041

Pınar Sude Gürdamar - 120203073

Fatma Hilal Börklü - 119200087



# Abstract 

This study aims to comprehensively analyze the factors influencing mobile app ratings. Understanding these elements is essential to improving user satisfaction and optimizing app performance. The study looks into the application of machine learning algorithms and how variables like user reviews, app cost, and genre affect app ratings. It also examines the emotional reactions of users both before and after the COVID-19 outbreak. In general, the study offers significant perspectives for marketers and app developers to enhance customer contentment and maximize app efficiency.


# Scope of the Project

This study intends to conduct a thorough investigation into the elements that influence mobile app evaluations. It focuses on such as user reviews, app pricing, and app genres to identify their proportional value in deciding app scores. The research will investigate several machine learning techniques to acquire a thorough understanding of the elements that influence app installation ratings. By combining features such as emotional content gathered from user reviews, pricing policies, and other external factors, the project hopes to create predictive models capable of accurately projecting app download numbers.  

Moreover, the study will investigate the emotional changes in user evaluations before and after the COVID-19 epidemic using particular sentiment analysis techniques and machine learning methods. By evaluating user attitudes across several life domains before and after the outbreak, the research hopes to capture users' nuanced emotional reactions and explore how these reactions are influenced by external events. The scope includes determining the most accurate machine learning and sentiment analysis methodologies for capturing these emotional shifts, as well as insights into the changing landscape of user preferences and behaviors in reaction to major events such as the COVID-19 epidemic.


# Research Questions

Which features play a more decisive role in influencing app scores: user reviews, app pricing, or app genre? Which features can be best utilized to enhance the accuracy of the score prediction model?

Which machine learning techniques may be used to fully comprehend the elements influencing the app installations' rating? How do these algorithms utilize features such as emotional content, pricing policies, user reviews, and other external factors to predict app installation numbers effectively?

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


# About Preprocessing Data

We have 4 datasets, includes Google Play Apps, their technical details and reviews in terms of before and after Covid-19 pandemic. 
In python file called ‘Ece.py', the main purpose of this code is to clean and preprocess data from the "Reviews.csv" and "Apps.csv" datasets. It includes operations including merging the two datasets, addressing missing values, transforming data types, combining datasets, deleting unnecessary columns, renaming columns, and standardizing specific data formats. For example, it accepts 'Varies with device', 'M', and 'k' values in the 'size' column, deletes the rows containing them,  transforms install counts to relevant float numbers, and replaces missing values with means or suitable placeholders. It also translates data types to acceptable representations. Also, unnecessary colums are removed to reduce crowding. All things considered, the code makes sure the data is in a consistent and useable format for additional modeling or analysis. 

On Emel_Preprocessed Parent dataset file, "googleplaystore.csv" and "googleplaystore_user_reviews.csv" datasets were merged according to our project's goal. Insufficient columns were identified and dropped whereas the rest were merged into one. Common applications from both datasets were chosen since the generated dataset aims to provide predictions using technical details and reviews and also make emotional analyses. In order to fit the data, the types were redefined. All the data below "Size" column were converted to bytes and the empty rows were filled by the mean value. In order to make sure that the data under "Size", "Installs" and "Price" columns were typed as float, the symbols were eliminated.

On Paired Parent&Child App Names file the unmatched apps from "googleplaystore.csv" and "Apps.csv" were dropped in order to analyze and see the technical details of these apps with adn without the effect of external factor (covid-19). 

# Gantt diagrams

## Syntax

```
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
    Handling Missing and Outlier Values         :active,  dp1, 2024-04-01, 2024-04-15
    Data Transformation and Normalization       :         dp2, 2024-04-01, 2024-04-15
    Feature Engineering and Encoding Techniques :         dp3, 2024-04-01, 2024-04-15

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

# Which features play a more decisive role in influencing app scores: user reviews, app pricing, or app genre? Which features can be best utilized to enhance the accuracy of the score prediction model?


When machine learning models are evaluated, especially when the results obtained from Random Forest , Linear Regression and Gradient Boosting Regressor models are examined, it is clearly seen that factors such as user comments (Reviews) and application type (Genres) affect application scores. It has been determined that especially in the Random Forest Regressor model, the "Reviews" and "Genres" features are decisive and these features are highly effective in score prediction. Therefore, user reviews and app type are among the most effective features for predicting app scores. These features can be optimally used to improve the accuracy of the score prediction model. User reviews are valuable because they provide direct feedback about the quality of the app and user satisfaction. Additionally, the app type also plays an important role in score estimation as it reflects the app's overall category and target audience. App pricing may also be effective in score estimation, but the effect of this factor was seen to be less pronounced compared to others in the models used in this study. While the Linear Regression model performed reasonably well with a mean square error of 0.0226 and an R-squared value of 0.5023 on the test set, the Gradient Boosting Regressor model achieved very low error rates and a high R-squared value of 0.9991, predicting application scores extremely accurately. It was seen that he did. Additionally, it was remarkable that the Random Forest Regressor model achieved accuracy scores of up to 100% in the training and test sets, showing that especially the "Reviews" (75.654%) and "Genres" (24.346%) features were decisive in score prediction. These results show that the most effective features for predicting app ratings are user reviews and app type.


# Emotional Studies: Before and After the Pandemic

The studies presented in the graphs on “SentimentAnalysisGraphs.ipynb” capture the changes in users' emotional reactions before and after the COVID-19 pandemic in several significant ways. Here's everything in detail:

Pre-COVID Sentiment Distribution: The pie chart indicates that before the pandemic, a majority of the sentiments were positive (65.1%), with negative sentiments at 26.1%, and neutral sentiments at 8.9%.
Post-COVID Sentiment Distribution: The second pie chart shows a slight decrease in positive sentiments to 61%, a small decrease in negative sentiments to 24.4%, and an increase in neutral sentiments to 14.6%.

![Ekran görüntüsü 2024-06-17 124639](https://github.com/BILGI-IE-423/ie423-2024-termproject-codecharmers/assets/159184426/0b46f365-c378-4ef4-958b-a819b8dd258f)
*Figure 1: Sentiment Distribution Pre-COVID-19*
![Ekran görüntüsü 2024-06-17 124647](https://github.com/BILGI-IE-423/ie423-2024-termproject-codecharmers/assets/159184426/34a469e9-4f1a-4e2f-9be4-2ef9a2fcfbf6)
*Figure 2: Sentiment Distribution Post-COVID-19*

The emotion distribution bar chart comparing pre-,and post-COVID reveals an obvious decrease in positive emotions and a little decline in negative emotions, with a notable increase in neutral sentiments following the COVID-19 pandemic. This implies that consumers' emotional responses during the epidemic were less judgmental or unsure.

![Ekran görüntüsü 2024-06-17 124615](https://github.com/BILGI-IE-423/ie423-2024-termproject-codecharmers/assets/159184426/a7438008-0c94-4303-b2d1-d1f638f19cd4)
*Figure 3: Sentiment Distribution Pre- and Post-COVID-19*

The line graph shows the evolution of sentiment trends over time; it shows an obvious decrease in positive emotions prior to COVID-19 and an increase in neutral feelings in response, while negative sentiments stay mostly unchanged. This pattern shows a change in user responses, which may have been impacted by the pandemic's stress and uncertainty on a worldwide scale.

![Ekran görüntüsü 2024-06-17 124703](https://github.com/BILGI-IE-423/ie423-2024-termproject-codecharmers/assets/159184426/d57afbe2-74bc-4a23-9da3-e15b7a9c401a)
*Figure 4: Sentiment Trends in Pre- and Post-COVID-19*

Pre-COVID Genre Usage: The bar chart shows high usage of Education and Photography apps before the pandemic, with other genres having significantly lower usage rates.
During COVID Genre Usage: There's a noticeable shift in genre usage during the pandemic, with Education and Photography remaining dominant, but there's a slight decrease in their proportion, indicating a diversification in app usage.

![Ekran görüntüsü 2024-06-17 132444](https://github.com/BILGI-IE-423/ie423-2024-termproject-codecharmers/assets/159184426/99834023-aa77-4f5d-8633-5adea258bc9e)
*Figure 5: Pre-COVID Genre Usage*
![Ekran görüntüsü 2024-06-17 124727](https://github.com/BILGI-IE-423/ie423-2024-termproject-codecharmers/assets/159184426/461a2d21-c1b9-4359-af50-c386203f9849)
*Figure 6: During COVID Genre Usage*

The comparison bar chart illustrates how consumption trends for different app genres changed. For instance, during the pandemic, there was a noticeable rise in the use of Tools and Art-Design apps, indicating changes in user requirements and activities. In contrast, Education and Photography apps continued to have significant utilization

![Ekran görüntüsü 2024-06-17 124739](https://github.com/BILGI-IE-423/ie423-2024-termproject-codecharmers/assets/159184426/2e554eb2-1dd1-4698-9950-949943f1c1ff)
*Figure 7: Genre Usage Comparision*

Before and during the epidemic, the stacked bar chart offers a thorough analysis of the sentiment distribution across several app categories. Although there is an obvious increase in neutral opinions across all categories throughout the pandemic, areas with significant positive sentiments include Tools, Health & Fitness, and Education. This shows the need to apps that are in Tools, Health & Fitness, and Education genres increased during pandemic. Also, these apps made lives easier and changed the dynamics of education, workout, and business life on daily basis. 

![Ekran görüntüsü 2024-06-17 124801](https://github.com/BILGI-IE-423/ie423-2024-termproject-codecharmers/assets/159184426/2fc3fbaf-74d4-498d-9994-294be79a2ccf)
*Figure 8: Sentiment Distribution by Category*



REFERENCES 

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

