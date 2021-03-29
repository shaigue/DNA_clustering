# TODO: add type annotations https://docs.python.org/3/library/typing.html where simple,
# TODO: add short docstrings for functions, for example:
#   def func(arg_1,...,arg_n):
#       """Calculates bla bla bla... and returns the final value"""
#       <code>
#   can add only a simple description, not full explanation of everything
#   see example in complex_generative_model.py


import random

def cluster(s, r, q, w, l, t_low, t_high, local_steps):
    c = s
    for i in range(0, local_steps):
        h_vals = [0] * len(c)
        s_w = sample_pi(w)[0]
        for k in range(0, len(c)):
            h_vals[k] = pi_func(c[k][0], s_w, w, l)
        for l_1 in range(0, len(c)):
            for l_2 in range(0, l_1):
                if h_vals[l_1] == h_vals[l_2]:
                    if l_2 >= len(c) or l_1 >= len(c):
                        break
                    if q_gram_dist(c[l_1][0], c[l_2][0], q) <= t_low or (q_gram_dist(c[l_1][0], c[l_2][0], q) <= t_high and edit_dist(c[l_1][0], c[l_2][0], len(c[l_1][0]), len(c[l_2][0]) <= r)):
                        union = c[l_1] + c[l_2]
                        c.pop(l_1)
                        c.pop(l_2)
                        c.append(union)
    return c

def edit_dist(a, b, l_a, l_b): #l_a = len(a), l_b = len(b)
    if l_a == 0:
        return l_b
    if l_b == 0:
        return l_a
    if a[l_a-1] == b[l_b-1]:
        return edit_dist(a, b, l_a-1, l_b-1)
    return 1 + min(edit_dist(a, b, l_a, l_b-1), edit_dist(a, b, l_a-1, l_b), edit_dist(a, b, l_a-1, l_b-1))

def pi_func(a, s_w, w, l):
    index = a.find(to_bin_str(s_w, w))
    if index < 0:
        return a
    return a[index: index+w+l]

def sample_pi(w):
    perm = [-1] * (2 ** w)
    index = list(range(0, 2 ** w))
    i = 2 ** w - 1
    while i >= 0:
        x = random.randint(0, len(index) - 1)
        perm[i] = index.pop(x)
        i -= 1
    return perm

def q_gram_dist(a, b, q):
    dist = 0
    a_count = [0] * (2 ** q)
    b_count = [0] * (2 ** q)
    index = [""] * (2 ** q)
    for i in range(0, 2 ** q):
        index[i] = to_bin_str(i, q)
        a_count[i] += a.count(index[i])
        b_count[i] += b.count(index[i])
        dist += abs(a_count[i]-b_count[i])
    return dist

def to_bin_str(x, n):
    t = ""
    for j in range(n - 1, -1, -1):
        d = (x >> j) % 2
        t += str(d)
    return t