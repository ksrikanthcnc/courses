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
distribution = set['Class'].value_counts().sortlevel()
# Visualize, can also check %s
distribution.plot(kind = 'bar')
plt.grid()

# Create a random returning model (worst case) for benchmarking
argmax() # ceiling

# Results
log_loss() # function to calculate error
C = confusion_matrix(y_true, y_pred)
sns.heatmap(C, annot=True, cmap='Y1GnBu', fmt=".3f")

# Plots
# Cumulative distribution
unique_genes = train_df['Gene'].value_counts()
plt.plot(
    np.cumsum(
        unique_genes.values / sum(unique_genes.values)
    ))
# Encoding (One-Hot, Response)
