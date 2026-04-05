# 📊 Data Normalization & Categorical Encoding in Python


---

## 🎯 Aim

To study and perform:

* Data normalization
* Conversion of categorical variables into quantitative variables

using Python functions and operations.

---

## 📘 Theory

In data analysis and machine learning, raw data often contains:

* Different numerical scales
* Categorical (non-numeric) values

Before analysis, data must be preprocessed.

### 🔑 Key Techniques:

* Data Normalization
* Categorical Variable Encoding

---

# 🔹 1. Data Normalization

## 📌 Definition

Scaling numerical data into a common range so all features contribute equally.

### 📊 Example

| Feature    | Range       |
| ---------- | ----------- |
| Price      | 500 – 50000 |
| Rating     | 1 – 5       |
| Units Sold | 10 – 1000   |

---

## ✅ Why Normalize?

* Improves model performance
* Prevents bias from large values
* Makes comparison easier

---

## 🔧 Types of Normalization

### 1. Min-Max Normalization

Scales values between **0 and 1**

```python
import pandas as pd

df['Price_MinMax'] = (df['Price'] - df['Price'].min()) / (df['Price'].max() - df['Price'].min())
print(df[['Price', 'Price_MinMax']])
```

---

### 2. Z-score Normalization

Centers data using mean & standard deviation

```python
df['Price_Zscore'] = (df['Price'] - df['Price'].mean()) / df['Price'].std()
print(df[['Price', 'Price_Zscore']])
```

---

### 3. Decimal Scaling

Divides values by powers of 10

```python
df['Price_Decimal'] = df['Price'] / 1000
print(df[['Price', 'Price_Decimal']])
```

---

## 🛠 Useful Functions

| Function           | Purpose            |
| ------------------ | ------------------ |
| `describe()`       | Summary statistics |
| `dtypes`           | Check data types   |
| `loc[]` / `iloc[]` | Select columns     |

---

# 🔹 2. Categorical to Quantitative Variables

## 📌 Definition

Converting non-numeric data into numeric form.

### 📊 Examples

| Variable         | Values                |
| ---------------- | --------------------- |
| Gender           | Male, Female          |
| Payment Method   | UPI, Debit Card       |
| Product Category | Electronics, Clothing |

---

## 🔄 Encoding Methods

### 1. Label Encoding

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Gender_Encoded'] = le.fit_transform(df['Gender'])

print(df[['Gender', 'Gender_Encoded']])
```

---

### 2. One-Hot Encoding

```python
df_onehot = pd.get_dummies(df, columns=['Product_Category'])
print(df_onehot.head())
```

---

### 3. Dummy Encoding

```python
df_dummy = pd.get_dummies(df, columns=['Product_Category'], drop_first=True)
print(df_dummy.head())
```

---

## 📌 Comparison

| Method           | Use Case                      |
| ---------------- | ----------------------------- |
| Label Encoding   | Ordered categories            |
| One-Hot Encoding | No order, multiple categories |
| Dummy Encoding   | Avoid multicollinearity       |

---

# 🧪 Sample Dataset Code

```python
import pandas as pd

data = {
    'Product': ['A', 'B', 'C'],
    'Price': [1000, 2000, 3000],
    'Category': ['Electronics', 'Clothing', 'Electronics']
}

df = pd.DataFrame(data)
print(df)
```

---

# ✅ Conclusion

* Data normalization ensures fair comparison between features
* Categorical encoding converts text data into usable numeric form
* These preprocessing steps improve model accuracy and efficiency

---

# ⭐ Author

**Ansh Yadav**

---

# 📌 Notes

* Always normalize before training ML models
* Use One-Hot Encoding for non-ordinal data
* Check data types before preprocessing
