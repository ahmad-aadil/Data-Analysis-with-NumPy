#  NumPy Data Analysis & Array Operations

This repository demonstrates hands-on data analysis using **NumPy**, one of Python’s most powerful numerical computing libraries.  
It walks through everything from creating arrays to performing mathematical, statistical, and broadcasting operations — with practical examples based on a real dataset (`forestfires.csv`).

---

##  Project Overview

In this project, we explore:
- Basic data loading and exploration with **Pandas**
- Creation and manipulation of **NumPy ndarrays**
- Understanding **NumPy data types and attributes**
- Array **reshaping**, **slicing**, and **indexing**
- Performing **arithmetic**, **statistical**, and **string** operations
- Exploring **broadcasting** and **random sampling**
- Implementing **vectorized math functions** for efficient computation

The project uses a dataset (`forestfires.csv`) to extract and manipulate variables related to environmental conditions such as temperature, humidity, wind, and rainfall.

---

##  Dataset

**Dataset:** [`forestfires.csv`](https://archive.ics.uci.edu/ml/datasets/Forest+Fires)  
**Source:** UCI Machine Learning Repository

The dataset includes attributes like:
- `temp` — temperature in Celsius
- `RH` — relative humidity (%)
- `wind` — wind speed (km/h)
- `rain` — rainfall (mm/m²)
- `area` — burned area of the forest (hectares)

Example preview:
```python
import pandas as pd

data = 'C:/datasets/forestfires.csv'
df = pd.read_csv(data)
df.head()
```

| X | Y | month | day | temp | RH | wind | rain | area |
|---|---|--------|-----|------|----|------|------|------|
| 7 | 5 | mar | fri | 8.2 | 51 | 6.7 | 0.0 | 0.0 |

---

##  NumPy in Action

### 1️ Creating Arrays
```python
import numpy as np

a = np.array([1, 2, 3])  # 1D array
b = np.array([[1, 2], [3, 4]])  # 2D array
c = np.array([[1, 2], [3, 4], [5, 6]])  # 3D-like array
```

### 2️ Working with the Forest Fires Data
```python
fires = df.iloc[:5, 8:]
print(fires)
```

| temp | RH | wind | rain | area |
|------|----|------|------|------|
| 8.2 | 51 | 6.7 | 0.0 | 0.0 |
| 18.0 | 33 | 0.9 | 0.0 | 0.0 |

Convert to NumPy:
```python
n1 = np.array(fires[:1])
n3 = np.array(fires[:3], dtype=int)
```

---

##  Array Operations

### Shape, Size & Dimensions
```python
print(n1.shape)  # (1, 5)
print(n3.size)   # 15
print(n3.ndim)   # 2
```

### Reshaping
```python
n3 = n3.reshape(5, 3)
```

### Broadcasting
```python
a = np.array([[10, 20, 30], [40, 50, 60]])
b = np.array([1, 2, 3])
print(a + b)
```

---

##  Statistical Functions

```python
arr = np.array([[5, 2, 3], [1, 5, 6], [3, 7, 4]])

print(np.amin(arr, axis=1))  # Min per row
print(np.amax(arr, axis=0))  # Max per column
print(np.mean(arr))          # Mean
print(np.var(arr))           # Variance
print(np.std(arr))           # Std deviation
```

---

##  Searching & Sorting

```python
data = np.array([[30, 40, 70], [80, 20, 10], [50, 90, 60]])

print(np.argmax(data))      # Index of max value
print(np.argmin(data))      # Index of min value
print(np.nonzero(data))     # Indices of non-zero elements
print(np.sort(data, axis=0))  # Column-wise sort
```

---

##  Random Sampling

```python
np.random.rand(3, 2)
np.random.randn(3)
np.random.randint(0, 10, size=(2, 4))
np.random.choice(5, 3)
```

---

##  Vectorized Math & Constants

```python
arr = np.array([1.56, 2.5, 3.6])
print(np.sqrt(arr))
print(np.round(arr))
print(np.floor(arr))
print(np.ceil(arr))
print(np.pi)
```

---

## 🧩 Fun Example — Building a Name from Arrays

```python
my_array = np.array(
    [
        [["A", "B", "C"], ["D", "E", "F"], ["G", "H", "I"]],
        [["J", "K", "L"], ["M", "N", "O"], ["P", "Q", "S"]],
        [["T", "U", "V"], ["W", "X", "Y"], ["Z", "AA", "AB"]],
    ]
)

my_name = (
    my_array[0,0,0]
    + my_array[0,0,0]
    + my_array[0,1,0]
    + my_array[0,2,2]
    + my_array[1,0,2]
)
print(my_name)  # Output: AADIL
```

---

##  Key Learnings

✅ How to work with **NumPy ndarrays**  
✅ Perform **array transformations** and **reshaping**  
✅ Understand **broadcasting rules**  
✅ Apply **statistical** and **arithmetic** functions efficiently  
✅ Use **vectorization** to replace loops  
✅ Work with **random number generation** for simulation or modeling

---

##  Requirements

Make sure you have the following installed:
```bash
pip install numpy pandas
```

---

##  Running the Code

Clone the repo and run the Jupyter notebook or Python script:

```bash
git clone https://github.com/yourusername/numpy-data-analysis.git
cd numpy-data-analysis
python numpy_analysis.py
```

---

##  License

This project is open-source and available under the **MIT License**.

---

## ✍️ Author

**Mr. Aadil**  
Data Analyst | AI | ML 


---


