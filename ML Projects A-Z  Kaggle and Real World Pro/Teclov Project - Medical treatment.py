# Outline
# Import modules

# Load
data_var = pd.read_csv('data.csv')
data_text = pd.read_csv('data_text.csv', sep="", names="")

# View
data_var.head(5)
data_var.info()
data_var.describe()
data_var.shape
data_var.columns
data_var.Class.unique()

# Errors are costly here

# Text Pre-Processing (NLTK)
# Stop words
# Numbers
# Special characters, extra spaces
# Lowercase

# Merge data_var and data_text
result = pd.merge(data_var, data_text, on = 'ID', how = 'left')

# Missing data
result[result.isnull().any(axis = 1)]
# handle missing data (here, just used Gene+Variation as Text if Nan)
result.loc[result[result.isnull(), 'TEXT'] = result['Gene'] + ' ' + result['Variation']]

# Split into Training, Cross-Validation and Test sets
# Check that the split is good (impartial)
distribution = df['Class'].value_counts().sortlevel()
# Visualize, can also check %s
distribution.plot(kind = 'bar')

# Create a random returning model (worst case) for benchmarking
np.random.rand(1,9)

# Results
log_loss() # function to calculate error
C = confusion_matrix(y_true, y_pred)
# Confusion matrix should have max diagonal values
sns.heatmap(C, annot=True, cmap='Y1GnBu', fmt=".3f")
# For precision and recall too

# Plots
unique_genes = train_df['Gene'].value_counts()
# Cumulative distribution
plt.plot(
    np.cumsum(
        unique_genes.values / sum(unique_genes.values)
    ))

# Encoding (One-Hot, Response)
# One - Hot
vec = CountVectorizer()
df_oh = vec.fit_transform(df['col'])
df_oh.shape
vec.get_feature_names()

# Laplace
clf = SGDClassifier(alpha=, penalti, loss)
clf.fit(one_hot)
sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
sig_clf.fit(one_hot)

# Overlap; Tells the importance of column
test_df[test_df['col'].isin(list(set(train_df['col'])))].shape[0]

12 ?