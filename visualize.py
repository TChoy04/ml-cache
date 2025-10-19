import numpy as np
import matplotlib.pyplot as plt
import random
from collections import OrderedDict, Counter, deque, defaultdict
from sklearn.linear_model import SGDClassifier
class LRUCache:
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.hits = 0
        self.accesses = 0
    def get(self, key):
        self.accesses += 1
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return True
        self.cache[key] = True
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return False
    def hit_rate(self):
        return self.hits / self.accesses if self.accesses else 0

class LFUCache:
    def __init__(self, capacity):
        self.cache = {}
        self.freq = Counter()
        self.capacity = capacity
        self.hits = 0
        self.accesses = 0

    def get(self, key):
        self.accesses += 1
        if key in self.cache:
            self.hits += 1
            self.freq[key] += 1
        else:
            if len(self.cache) >= self.capacity:
                lfu_key = min(self.freq, key=self.freq.get, default=None)
                if lfu_key:
                    self.cache.pop(lfu_key, None)
                    self.freq.pop(lfu_key, None)
            self.cache[key] = True
            self.freq[key] = 1
        return key in self.cache

    def hit_rate(self):
        return self.hits / self.accesses if self.accesses else 0

class ARCCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.t1, self.t2 = deque(), deque()
        self.b1, self.b2 = deque(), deque()
        self.p = 0
        self.hits = 0
        self.accesses = 0

    def replace(self, key):
        if len(self.t1) > 0 and ((key in self.b2 and len(self.t1) == self.p) or len(self.t1) > self.p):
            self.b1.append(self.t1.popleft())
        else:
            self.b2.append(self.t2.popleft())

    def get(self, key):
        self.accesses += 1
        if key in self.t1:
            self.hits += 1
            self.t1.remove(key)
            self.t2.append(key)
        elif key in self.t2:
            self.hits += 1
            self.t2.remove(key)
            self.t2.append(key)
        elif key in self.b1:
            self.p = min(self.capacity, self.p + max(len(self.b2) / len(self.b1), 1))
            self.replace(key)
            self.b1.remove(key)
            self.t2.append(key)
        elif key in self.b2:
            self.p = max(0, self.p - max(len(self.b1) / len(self.b2), 1))
            self.replace(key)
            self.b2.remove(key)
            self.t2.append(key)
        else:
            if len(self.t1) + len(self.b1) == self.capacity:
                if len(self.t1) < self.capacity:
                    self.b1.popleft()
                    self.replace(key)
                else:
                    self.t1.popleft()
            elif len(self.t1) + len(self.b1) < self.capacity and len(self.t1) + len(self.t2) + len(self.b1) + len(self.b2) >= self.capacity:
                if len(self.t1) + len(self.t2) + len(self.b1) + len(self.b2) == 2 * self.capacity:
                    self.b2.popleft()
                self.replace(key)
            self.t1.append(key)
        return key in self.t1 or key in self.t2

    def hit_rate(self):
        return self.hits / self.accesses if self.accesses else 0

class MLCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.recency = defaultdict(int)
        self.freq = defaultdict(int)
        self.time = 0
        self.hits = 0
        self.accesses = 0
        self.model = SGDClassifier(loss="log_loss")
        self.trained = False
        self.features, self.labels = [], []

    def _extract_features(self, key):
        return np.array([self.time - self.recency[key], self.freq[key]])

    def get(self, key):
        self.time += 1
        self.accesses += 1
        if key in self.cache:
            self.hits += 1
        reused = key in self.cache
        self.freq[key] += 1
        self.recency[key] = self.time
        self.features.append(self._extract_features(key))
        self.labels.append(int(reused))
        if len(self.features) >= 100 and not self.trained:
            self.model.partial_fit(self.features, self.labels, classes=[0, 1])
            self.trained = True
            self.features, self.labels = [], []
        elif len(self.features) >= 100:
            self.model.partial_fit(self.features, self.labels)
            self.features, self.labels = [], []
        if reused:
            return True
        if len(self.cache) >= self.capacity:
            predictions = {k: self._predict_reuse(k) for k in self.cache}
            to_evict = min(predictions, key=predictions.get)
            del self.cache[to_evict]
        self.cache[key] = True
        return False

    def _predict_reuse(self, key):
        if not self.trained:
            return random.random() * 0.5
        features = self._extract_features(key).reshape(1, -1)
        return self.model.predict_proba(features)[0][1]

    def hit_rate(self):
        return self.hits / self.accesses if self.accesses else 0
    
def generate_data(n_items=1000, n_accesses=100000, alpha=1.2, flips=[25000, 50000, 75000]):
    ranks = np.arange(1, n_items + 1)
    probs = 1 / np.power(ranks, alpha)
    probs /= probs.sum()
    trace = np.random.choice(ranks, size=n_accesses, p=probs)
    for f in flips:
        np.random.shuffle(trace[f:f + 5000])
    return trace

def run_all_caches():
    n_accesses = 100000
    trace = generate_data(n_accesses=n_accesses)
    policies = {"LRU": LRUCache(100), "LFU": LFUCache(100), "ARC": ARCCache(100), "MLCache": MLCache(100)}
    results = {name: [] for name in policies}
    window = 10000
    for i, key in enumerate(trace):
        for name, cache in policies.items():
            cache.get(key)
            if i % window == 0 and i > 0:
                results[name].append(cache.hit_rate())
    min_len = min(len(v) for v in results.values())
    for k in results:
        results[k] = np.array(results[k][:min_len])
        results[k] = (results[k] - results[k].min()) / (results[k].max() - results[k].min() + 1e-8)
    for name, data in results.items():
        plt.plot(data, label=name)
    plt.legend()
    plt.title("Cache Performance Comparison")
    plt.xlabel("Cache Calls (x10,000)")
    plt.ylabel("Normalized Hit Rate")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_all_caches()
