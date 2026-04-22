
predict = lambda x, w, b: 1 if sum(wi*xi for wi,xi in zip(w,x))+b >= 0 else 0

print(predict([1, 0], [0.5, -0.3], 0))



# **`lambda x, w, b:`**
# defines an anonymous function that takes three arguments:
# - `x` — the input vector (e.g. `[1, 0]`)
# - `w` — the weights vector (e.g. `[0.5, -0.3]`)
# - `b` — the bias (a single number)

# ---

# **`zip(w, x)`**
# pairs each weight with its corresponding input:
# ```
# w=[0.5, -0.3], x=[1, 0]  →  [(0.5, 1), (-0.3, 0)]
# ```

# ---

# **`wi*xi for wi,xi in zip(w,x)`**
# multiplies each pair — this is the element-wise multiplication of the two vectors:
# ```
# 0.5*1, -0.3*0  →  0.5, 0.0
# ```

# ---

# **`sum(...)`**
# adds them all up — this is the **dot product** of `w` and `x`:
# ```
# 0.5 + 0.0  →  0.5
# ```

# ---

# **`+ b`**
# adds the bias, shifting the result:
# ```
# 0.5 + b 