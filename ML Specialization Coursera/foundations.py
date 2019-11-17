import graphlab #scalable than sklearn

#
graphlab.canvas.set_target()

# 02_regression-predicting-house-prices
sf = graphlab.SFrame("data.csv")
sf.show()
sf[''].show(view='')
sf.random_split()
model=graphlab.linear_regression.create()
model.evaluate()
# matplotlib
model.get()

# 03_classification-analyzing-sentiment
graphlab.text_analytics.count_words()
model=graphlab.logistic_classifier.create()

# 04_clustering-and-similarity-retrieving-documents
# tf-idf
graphlab.text_analytics.tf_idf()
graphlab.distances.cosine()
model=graphlab.nearest_neighbors.create()
model.query()

# 05_recommending-products
model=graphlab.popularity_recommender.create()
model.recommend()
model=graphlab.item_similarity.create()
model.recommend()
model.get_similar_items()
# matplotlib
performance=graphlab.recommender.util.compare_models()

# 06_deep-learning-searching-for-images
model=graphlab.load_model()
'features'=model.extract_features()
model=graphlab.logistic_classifier.create()
model=graphlab.nearest_neighbors.create()

# 07_closing-remarks
# 08_Resources
