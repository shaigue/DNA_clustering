import random
from typing import List


def cluster(s: List[List[str]],
            r: int, q: int, w: int, l: int,
            t_low: int, t_high: int, local_steps: int) \
        -> List[List[str]]:
    """ Main clustering function similar to described article algorithm,
    excluding multi-threading part. 
    Returning the final algorithm clustering. """
    
    c = s
    for i in range(0, local_steps):
        h_vals = [""] * len(c)
        s_w = sample_pi(w)[0]
        for k in range(0, len(c)):
            h_vals[k] = pi_func(c[k][0], s_w, w, l)
        for l_1 in range(0, len(c)):
            for l_2 in range(0, l_1):
                if h_vals[l_1] == h_vals[l_2]:
                    if l_2 >= len(c) or l_1 >= len(c):
                        break
                    if q_gram_dist(c[l_1][0], c[l_2][0], q) <= t_low or (
                            q_gram_dist(c[l_1][0], c[l_2][0], q) <= t_high and edit_dist(c[l_1][0], c[l_2][0],
                                                                                         len(c[l_1][0]),
                                                                                         len(c[l_2][0]) <= r)):
                        union = c[l_1] + c[l_2]
                        c.pop(l_1)
                        c.pop(l_2)
                        c.append(union)
    return c


def edit_dist(a: str, b: str, l_a: int, l_b: int) -> int:
    """ l_a = len(a), l_b = len(b).
    Simple edit distance calculating function, using recursion. """

    if l_a == 0:
        return l_b
    if l_b == 0:
        return l_a
    if a[l_a - 1] == b[l_b - 1]:
        return edit_dist(a, b, l_a - 1, l_b - 1)
    return 1 + min(edit_dist(a, b, l_a, l_b - 1), edit_dist(a, b, l_a - 1, l_b), edit_dist(a, b, l_a - 1, l_b - 1))


def pi_func(a: str, s_w: int, w: int, l: int) -> str:
    """ Used to initialize h_vals, the hash values for each cluster.
    w,l parameters as described in the algorithm - cutting the representative string, returns the cutted string. """

    index = a.find(to_bin_str(s_w, w))
    if index < 0:
        return a
    return a[index: index + w + l]


def sample_pi(w: int) -> List[int]:
    """ Creating a permutation of (2**w) numbers.
     Used later for one random number in [1, 2**w]. """

    perm = [-1] * (2 ** w)
    index = list(range(0, 2 ** w))
    i = 2 ** w - 1
    while i >= 0:
        x = random.randint(0, len(index) - 1)
        perm[i] = index.pop(x)
        i -= 1
    return perm


def q_gram_dist(a: str, b: str, q: int) -> int:
    """ Calculating the q_gram distance between two strings as described in the article. """

    dist = 0
    a_count = [0] * (2 ** q)
    b_count = [0] * (2 ** q)
    index = [""] * (2 ** q)
    for i in range(0, 2 ** q):
        index[i] = to_bin_str(i, q)
        a_count[i] += a.count(index[i])
        b_count[i] += b.count(index[i])
        dist += abs(a_count[i] - b_count[i])
    return dist


def to_bin_str(x: int, n: int) -> str:
    """ Translates a positive number to a n-length binary number. """

    t = ""
    for j in range(n - 1, -1, -1):
        d = (x >> j) % 2
        t += str(d)
    return t
