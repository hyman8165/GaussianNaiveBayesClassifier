import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class NaiveBayesClassifier:
  def __init__(self):
    self.features = list
    self.likelihoods = {}
    self.class_priors = {}
    self.pred_priors = {}

    self.train_size = int
    self.num_feats = int
    self.outcomes = np.array

  def fit(self, X, y):
    self.features = list(X.columns)
    self.train_size = X.shape[0]
    self.num_feats = X.shape[1]
    self.outcoms = np.unique(y)

    for feature in self.features:
      self.likelihoods[feature] = {}
      self.pred_priors[feature] = {}
      for outcome in self.outcoms:
        mean = np.mean(X[feature][y == outcome])
        std = np.std(X[feature][y == outcome])
        self.likelihoods[feature][outcome] = {'mean': mean, 'std': std}
    for outcome in self.outcoms:
      outcome_count = sum(y == outcome)
      self.class_priors[outcome] = outcome_count / self.train_size

  def predict(self, X):
    results = []
    for query in np.array(X):
      probs_outcome = {}
      for outcome in self.outcoms:
        prior = self.class_priors[outcome]
        likelihood = 1
        for feature, feat_val in zip(self.features, query):
          mu = self.likelihoods[feature][outcome]['mean']
          sigma = self.likelihoods[feature][outcome]['std']
          likelihood *= np.exp(-1*(feat_val-mu)**2/(2*sigma**2))/(np.sqrt(np.pi*2)*(sigma**2))
        posterior = likelihood * prior
        probs_outcome[outcome] = posterior
      result = max(probs_outcome, key = lambda x: probs_outcome[x])
      results.append(result)
    return results
  
mean0 = [-1.8, -9.5]
cov0 = [[1.8**2, 0], [0, 1.8**2]]
mean1 = [1.2, -1]
cov1 = [[2.2**2, 0], [0, 1.8**2]]
rng = np.random.RandomState(42)
data0 = rng.multivariate_normal(mean0, cov0, 200)
data1 = rng.multivariate_normal(mean1, cov1, 200)
dataX = np.vstack((data0, data1))
y0 = np.zeros(200)
y1 = np.ones(200)
y = np.hstack((y0, y1)).astype(int)
testIndx = rng.choice(400, size=80, replace=False, p=None)
df = pd.DataFrame(dataX, columns=['X0', 'X1'])
df['y'] = y
df_test = df.take(testIndx)
df_train = df.drop(index=testIndx)

fig, ax = plt.subplots(1, 2, figsize=(12,6))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
ax[0].scatter(x='X0', y='X1', c='y', data=df_train)
ax[0].title.set_text('Train set')
ax[1].scatter(x='X0', y='X1', c='y', data=df_test)
ax[1].title.set_text('Test set')
plt.show()

nbc = NaiveBayesClassifier()
X_train = df_train.drop('y', axis=1)
y_train = df_train['y']
nbc.fit(X_train, y_train)

X_test = df_test.drop('y', axis=1)
y_test = df_test['y']
yhat = nbc.predict(X_test)
auc = np.mean(yhat==y_test)
print(auc)

x_min, x_max = df['X0'].min() - 0.5, df['X0'].max() + 0.5
y_min, y_max = df['X1'].min() - 0.5, df['X1'].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
z = nbc.predict(np.c_[xx.ravel(), yy.ravel()])
z = np.array(z).reshape(xx.shape)
plt.pcolormesh(xx, yy, z, cmap=plt.cm.Paired)
plt.scatter(x='X0', y='X1', c='y', data=df)
plt.show()
