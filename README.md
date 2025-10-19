# Machine Learning Cache
A machine learning cache policy, comparing performance against traditional algorithms (LRU, LFU, ARC) under Zipfian workloads using Python and scikit-learn.
</br>
</br>
We can observe from the graphs below that the traditional caching policies generally hit relatively stable hit rates under Zipfian workloads which is expected due to their skewed access patterns. However, the machine learning cache shows much higher variance, sometimes initially matching or surpassing other policies but can experience sharp drops in hit rate. These fluctuations suggest that while the machine learning cache can learn and exploit short term access patterns, it is extremely sensitive to change. 

# Samples:
![first graph](./graphs/graph1.png)
![second graph](./graphs/graph2.png)
![third graph](./graphs/graph3.png)


## run pip install on the following libraries if not present:
numpy
matplotlib
scikit-learn

## running the program
Run: ```python visualize.py```

Will likely take many seconds to finish running since it'll run on your cpu

