from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def prepare_data(input_df, categories):
    x = input_df.iloc[:, :11].astype(float)
    y = input_df[['expert_consensus']].to_numpy().flatten()

    le = LabelEncoder()
    le.fit(categories)
    y = le.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=68)

    return X_train, X_test, y_train, y_test, le
