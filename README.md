PART 1 ANSWERS
Q1: Explain the primary differences between TensorFlow and PyTorch. When would you choose one over the other?
Feature	TensorFlow	PyTorch
Computational Graph	Uses static graphs (TensorFlow 1.x) and dynamic graphs via eager execution (in newer versions).	Uses dynamic computation graphs by default.
Syntax	More structured and production-oriented.	More Pythonic and intuitive for research.
Deployment	Integrated with TensorFlow Serving, TensorFlow Lite, and TensorFlow.js.	Uses TorchServe, but deployment features are less extensive.
Community Usage	Popular in enterprise and large-scale production.	Widely preferred in academic research and experimentation.
When to choose TensorFlow:
•	For deploying models in production environments.
•	When targeting mobile/edge devices.
•	When using complex model-serving infrastructure.
 When to choose PyTorch:
•	For rapid prototyping and experimentation.
•	In academic research where flexibility is key.
•	When a more intuitive debugging and Python-like environment is preferred.
Q2: Describe two use cases for Jupyter Notebooks in AI development.
1.	Interactive Model Prototyping
o	Allows developers to write, test, and visualize code blocks step-by-step, ideal for building and debugging machine learning models.
2.	Data Exploration and Visualization
o	Jupyter Notebooks support inline charts and visualizations, making it easy to clean, analyze, and interpret datasets before training models.
Q3: How does spaCy enhance NLP tasks compared to basic Python string operations?
•	Advanced Linguistic Capabilities:
spaCy provides tokenization, part-of-speech tagging, named entity recognition (NER), and dependency parsing, which basic string operations cannot perform.
•	Optimized for Speed and Accuracy:
Unlike plain string methods, spaCy uses pre-trained statistical models and optimized Cython code, enabling efficient processing of large-scale text data with semantic understanding.
2. Comparative Analysis: Scikit-learn vs TensorFlow
Criteria	Scikit-learn	TensorFlow
Target Applications	Best for classical machine learning such as regression, clustering, and decision trees.	Designed for deep learning and neural networks, including CNNs, RNNs, and transformers.
Ease of Use	Very beginner-friendly with simple API and minimal code requirements.	Steeper learning curve due to advanced features and larger framework.
Community Support	Strong support in traditional ML communities and academia.	Extensive community, backed by Google, with large industry adoption for AI applications.

