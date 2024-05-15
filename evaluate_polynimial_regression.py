from utils import *

# Generate some data
X,y,x_ideal,y_ideal = gen_data(18, 2, 0.7)
# X.shape (18,) y.shape (18,)

""" Split the data using sklearn train_test_split """
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# X_train.shape (12,) y_train.shape (12,)
# X_test.shape (6,) y_test.shape (6,)

    