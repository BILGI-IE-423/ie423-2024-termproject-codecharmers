[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/N24Xct0L)

Research Questions

Which features play a more decisive role in influencing app scores: user reviews, app pricing, or app genre? Which features can be best utilized to enhance the accuracy of the score prediction model?

Which machine learning techniques may be used to fully comprehend the elements influencing the app installations' rating? How do these algorithms utilize features such as emotional content, pricing policies, user reviews, and other external factors to predict app installation numbers effectively?

Considering the emotional changes in user evaluations before to and during COVID-19, what particular machine learning and sentiment analysis approaches can be used? In what ways do these studies most accurately capture the changes in users' emotional reactions before and after the epidemic in various spheres of life?


RELATED WORKS

The internet's growth has allowed users to share their opinions on social media and commercial websites. As a result, analyzing sentiments and emotions in text has become an important area of research. This field focuses on automatically classifying user reviews to gain insights into public sentiment.

Researchers usually classify customer reviews into three categories: positive, negative, or neutral. But, since reviews can be super positive or super negative, using a specific scale to measure how positive or negative they are could make sentiment analysis work better. (Singh et al., 2016).

Within the field of text classification, researchers have developed several techniques. For example, Dai et al. (2007) used an iterative Expectation-Maximization algorithm to transfer a Naïve Bayes classifier from one domain to another. This method allowed them to apply the classifier to a new context effectively. In their 2008 study, Gao et al. employed a combination of multiple classifiers trained on various source domains to classify target documents by assessing their similarity to a clustering of the target documents.

In cross-domain classification or domain adaptation, text categorization is essential. It was Pan and Yang (2010) who suggested this concept where knowledge can be transferred between two domains that have different distributions but the same labels.

Techniques like Latent Dirichlet Allocation (Blei et al., 2003) or Latent Semantic Indexing (Weigend et al., 1999) to uncover hidden correlations among words, thereby enhancing document representations. More recent approaches extract semantic information from terms by utilizing external knowledge bases such as WordNet (Scott and Matwin, 1998) or Wikipedia (Gabrilovich and Markovitch, 2007).


One way to perform sentiment analysis is by the use of lexicons, which assign a sentiment value to each term. To enhance accuracy in calculating sentiment values from basic summation and mean methods, Jurek et al. (2015) suggested a normalization function. Lexicon-based methods can be categorized into two types: dictionary-based and corpus-based approaches.

In dictionary-based methods, a list of initial words is created and then expanded with words that have similar or opposite meanings (Schouten and Frasincar, 2015). On the other hand, corpus-based methods involve identifying sentiment words that are specific to a particular subject based on their usage in context (Bernabé-Moreno et al., 2020). Another approach suggested by Cho et al. (2014) is a three-step method to improve how polarity (positive or negative sentiment) is determined based on context, as well as making dictionaries more adaptable for different domains.

Another approach to sensitivity analysis involves machine learning algorithms, where data sets are classified into training and test sets for model training and analysis Supervised classification algorithms such as Naïve Bayes, Support Vector Machine (SVM) are often used ), and decision trees are used (Gamon, 2004 ). Bučar et al. (2018) developed a sentiment lexicon and labeled news corpora to analyze sentiments in Slovene texts. They found that Naïve Bayes performed better than SVM. Tiwari et al. (2020) used SVM, Naïve Bayes, and maximum entropy algorithms with n-gram feature extraction on a dataset of movie reviews. They observed that accuracy decreased as the n-gram values increased.


REFERENCES 

Bernabé-Moreno J, Tejeda-Lorente A, Herce-Zelaya J, Porcel C, Herrera-Viedma E (2020) A context-aware embeddings supported method to extract a fuzzy sentiment polarity dictionary. Knowledge-Based Systems 190:105236.

Blei, D. M., Ng, A. Y., and Jordan, M. I. (2003). Latent Dirichlet allocation. The Journal of Machine Learning research, 3:993–1022.

Bučar, Jože & Žnidaršič, Martin & Povh, Janez. (2018). Annotated news corpora and a lexicon for sentiment analysis in Slovene. Language Resources and Evaluation. 52. 10.1007/s10579-018-9413-3.

Cho H, Kim S, Lee J, Lee JS (2014) Data-driven integration of multiple sentiment dictionaries for lexicon-based sentiment classification of product reviews. Knowledge-Based Systems 71:61–71.

Dai, W., Xue, G.-R., Yang, Q., and Yu, Y. (2007). Transferring naive bayes classifiers for text classification. In Proceedings of the AAAI ’07, 22nd national conference on Artificial intelligence, pages 540–545.

Gabrilovich, E. and Markovitch, S. (2007). Computing semantic relatedness using Wikipedia-based explicit semantic analysis. In Proceedings of the 20th International Joint Conference on Artificial Intelligence, volume 7, pages 1606–1611.

Gamon M (2004) Sentiment classification on customer feedback data: noisy data, large feature vectors, and the role of linguistic analysis. In: COLING 2004: Proceedings of the 20th international conference on computational linguistics, pp 841–847.

Jurek A, Mulvenna MD, Bi Y (2015) Improved lexicon-based sentiment analysis for social media analytics. Security Inform 4(1):1–13.

Jurek-Loughrey, Anna & Mulvenna, Maurice & Bi, Yaxin. (2015). Improved lexicon-based sentiment analysis for social media analytics. Security Informatics. 4. 10.1186/s13388-015-0024-x.

Pan, S. J., Kwok, J. T., and Yang, Q. (2008). Transfer learning via dimensionality reduction. In Proceedings of the AAAI ’08, 23rd national conference on Artificial intelligence, pages 677–682.

Sam Scott and Stan Matwin. 1998. Text Classification Using WordNet Hypernyms. In Usage of WordNet in Natural Language Processing Systems.

Schouten K, Frasincar F (2015) Survey on aspect-level sentiment analysis. IEEE Trans Knowl Data Eng 28(3):813–830.

Schouten, Kim & Frasincar, Flavius. (2015). Survey on Aspect-Level Sentiment Analysis. IEEE Transactions on Knowledge and Data Engineering. 28. 1-1. 10.1109/TKDE.2015.2485209.

Scott, S. and Matwin, S. (1998). Text classification using WordNet hypernyms. In Use of WordNet in natural language processing systems: Proceedings of the conference, pages 38–44.

Singh, Mangal & Nafis, Md Tabrez & Mani, Neel. (2016). Sentiment Analysis and Similarity Evaluation for Heterogeneous-Domain Product Reviews. International Journal of Computer Applications. 144. 16-19. 10.5120/ijca2016910112.

Tiwari P, Mishra BK, Kumar S, Kumar V (2020) Implementation of n-gram methodology for rotten tomatoes review dataset sentiment analysis. In: Cognitive analytics: concepts, methodologies, tools, and applications, IGI Global, pp 689–701.

Tiwari P, Mishra BK, Kumar S, Kumar V (2020) Implementation of n-gram methodology for rotten tomatoes review dataset sentiment analysis. In: Cognitive analytics: concepts, methodologies, tools, and applications, IGI Global, pp 689–701.

Weigend, A. S., Wiener, E. D., and Pedersen, J. O. (1999). Exploiting hierarchy in text categorization. Information Retrieval, 1(3):193–216.

S. J. Pan and Q. Yang, "A Survey on Transfer Learning," in IEEE Transactions on Knowledge and Data Engineering, vol. 22, no. 10, pp. 1345-1359, Oct. 2010, doi: 10.1109/TKDE.2009.191.
