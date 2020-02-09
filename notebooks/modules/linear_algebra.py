#!/usr/bin/env python
# coding: utf-8

# # Vectors

# In[2]:


from typing import List

Vector = List[float]


# (In the following we are using the [zip](https://docs.python.org/3.3/library/functions.html#zip)-functionto make an iterator that aggregates elements from each of the iterables.)

# ## Addition

# In[3]:


def add(v: Vector, w: Vector) -> Vector:
    """Adds corresponding elements"""
    assert len(v) == len(w), "vectors must be the same length"
    return [v_i + w_i for v_i, w_i in zip(v, w)]
assert add([1, 2, 3], [4, 5, 6]) == [5, 7, 9]


# ## Subtraction

# In[4]:


def subtract(v: Vector, w: Vector) -> Vector:
    """Subracts corresponding elements"""
    assert len(v) == len(w), "vectors must be the same length"
    return [v_i - w_i for v_i, w_i in zip(v, w)]
assert subtract([5, 7, 9], [4, 5, 6]) == [1, 2, 3]


# ## Vector sum

# In[5]:


def vector_sum(vectors: List[Vector]) -> Vector:
    """Sums all correspondig elemements"""
    # Check that vectors is not empty
    assert vectors, "no vectors provided"
    
    # Check the vectors are all the same size
    num_elements = len(vectors[0])
    assert all(len(v) == num_elements for v in vectors), "different sizes!"
    
    # the i-th element of the result is the sum of every vector[i]
    return [sum(vector[i] for vector in vectors)
            for i in range(num_elements)]

assert vector_sum([[1, 2], [3, 4], [5, 6], [7, 8]]) == [16, 20]


# ## Scalar

# In[8]:


def scalar_multiply(c: float, v: Vector) -> Vector:
    """Multiplies every element by c"""
    return[c * v_i for v_i in v]

assert scalar_multiply(2, [1, 2, 3]) == [2, 4, 6]


# ## Vector mean

# In[9]:


def vector_mean(vectors: List[Vector]) -> Vector:
    """Computes the element-wise average"""
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))

assert vector_mean([[1, 2], [3, 4], [5, 6]]) == [3, 4]


# ## Dot product
# _The sum of their components products_ 

# In[12]:


def dot(v: Vector, w: Vector) -> float:
    """Computes v_1 * w_1 + ... + v_n * w_n"""
    assert len(v) == len(w), "vectors must be the same length"
    
    return sum(v_i * w_i for v_i, w_i in zip(v, w))

assert dot([1, 2, 3], [4, 5, 6]) == 32 # 1 * 4 + 2 * 5 + 3 * 6


# ## Sum of squares

# In[13]:


def sum_of_squares(v: Vector) -> float:
    """Returns v_1 * v_1 + ... + v_n * v_n"""
    return dot(v, v)

assert sum_of_squares([1, 2, 3]) == 14 # 1 * 1 + 2 * 2 + 3 * 3


# ## Magnitude (Length)

# In[14]:


import math

def magnitude(v: Vector) -> float:
    """Returns the magnitude (or length) of v"""
    return math.sqrt(sum_of_squares(v)) # math.sqrt is square root function

assert magnitude([3, 4]) == 5


# ## Distance

# In[23]:


def squared_distance(v: Vector, w: Vector) -> float:
    """Computes (v_1 - w_1) ** 2 + ... + (v_n - w_n) ** 2"""
    return sum_of_squares(subtract(v, w))

def distance(v: Vector, w: Vector) -> float:
    """Computes the distance between v and w"""
    return math.sqrt(squared_distance(v, w))

assert distance([1, 2, 3], [1, 2, 3]) == 0


# # Matrices

# In[24]:


# Another type alias
Matrix = List[List[float]]


# ## Shape

# In[28]:


from typing import Tuple

def shape(A: Matrix) -> Tuple[int, int]:
    """Returns (# of rows of A, # of columns of A)"""
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0 # number of elements in first row
    return num_rows, num_cols

assert shape([[1, 2, 3], [4, 5, 6]]) == (2,3) # 2 rows, 3 columns


# In[29]:


def get_row(A: Matrix, i: int) -> Vector:
    """Returns the i-th row of A (as a Vector)"""
    return A[i]                     # A[i] is already the ith row

def get_column(A: Matrix, j: int) -> Vector:
    """Returns the j-th column of A (as a Vector)"""
    return [A_i[j]                 # jth element of row A_i
            for A_i in A]          # for each row A_i


# In[30]:


from typing import Callable

def make_matrix(num_rows: int,
               num_cols: int,
               entry_fn: Callable[[int, int], float]) -> Matrix:
    """
    Returns a num_rows x num_cols matrix
    whose (i,j)-th entry is entry_fn(i,j)
    """
    return[[entry_fn(i, j)              # given i, create a list
            for j in range(num_cols)]   #   [entry_fn(i, 0), ...]
           for i in range(num_rows)]    # create on list for each i


# In[33]:


def identity_matrix(n: int) -> Matrix:
    """Returns the n x n identity matrix"""
    return make_matrix(n, n, lambda i, j: 1 if i == j else 0)

assert identity_matrix(5) == [[1, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0],
                              [0, 0, 1, 0, 0],
                              [0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 1]]


# ## Uses of matrices
# 1. Use a matrix to represent a dataset consisting of multiple vectors
# 2. Use an n x k matrix to represent a inear function that maps k-dimensional vectors to n-dimensional vectors
# 3. Use a matrix to represent binary relationships
