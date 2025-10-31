# Text Clustering and Visualization with t-SNE

This lab demonstrates text analysis techniques using embeddings, dimensionality reduction, and clustering to uncover patterns in textual data. The notebook provides a hands-on approach to understanding how transformer-based encoder embeddings can transform unstructured text into meaningful representations.

## 📚 Project Overview

The **Text Clustering and Visualization with t-SNE** notebook is a comprehensive exploration of text embedding techniques, dimensionality reduction, and unsupervised learning methods. It's designed to help learners understand how to:

- Convert text documents into numerical representations (embeddings)
- Apply dimensionality reduction techniques for visualization
- Perform clustering analysis on text data
- Visualize and interpret the results

## 🎯 Learning Objectives

By working through this notebook, you will learn to:

1. **Load and Preprocess Text Data** - Work with pre-chunked text documents from different subjects
2. **Generate Text Embeddings** - Use sentence transformer models to convert text to vectors
3. **Apply Dimensionality Reduction** - Use t-SNE to visualize high-dimensional embeddings in 2D space
4. **Perform Clustering Analysis** - Apply K-means clustering to group similar text chunks
5. **Visualize Results** - Create informative plots that reveal patterns in the data

## 🔬 Technical Approach

### Text Embeddings
The notebook uses the **BAAI/bge-base-en-v1.5** model, a powerful sentence transformer that converts text into 768-dimensional vectors. These embeddings preserve semantic relationships between words, phrases, and documents, making them ideal for clustering and classification tasks.

### Dimensionality Reduction with t-SNE
**t-SNE (t-Distributed Stochastic Neighbor Embedding)** is used to reduce the 768-dimensional embeddings to 2D for visualization. This technique:
- Preserves local similarities between data points
- Reveals clusters and patterns that might not be visible in high-dimensional space
- Uses perplexity parameters to balance local vs. global structure preservation

### Clustering with K-Means
**K-Means clustering** is applied to group similar text chunks together:
- Automatically identifies natural groupings in the data
- Provides cluster centers for interpretation
- Helps validate the quality of the embeddings

## 📊 Dataset

The notebook works with a curated dataset containing:
- **25,600+ text chunks** from textbooks belonging to three different subjects
- **Three subject domains**: History, Physics, and Biology
- **Pre-processed chunks** with minimum character length filtering (500+ characters)
- **Structured format** with labels, titles, and chunk indices

## 🛠️ Key Libraries and Dependencies

### Core ML Libraries
- **PyTorch** - Deep learning framework for tensor operations
- **SentenceTransformers** - Pre-trained transformer models for embeddings
- **scikit-learn** - Machine learning algorithms (t-SNE, K-Means)

### Data Processing
- **HuggingFace Datasets** - Efficient dataset loading and manipulation
- **NumPy** - Numerical computing and array operations

### Visualization
- **Matplotlib** - Basic plotting capabilities
- **Seaborn** - Statistical visualization and styling

## 🚀 Workflow Overview

1. **Data Loading** → Load pre-chunked text documents
2. **Text Filtering** → Remove chunks below minimum length threshold
3. **Embedding Generation** → Convert text to 768-dimensional vectors
4. **Dimensionality Reduction** → Apply t-SNE to get 2D coordinates
5. **Clustering** → Perform K-means clustering on reduced vectors
6. **Visualization** → Create scatter plots showing clusters and relationships

## 📈 Key Insights and Applications

### What You'll Discover
- **Semantic Clustering**: How similar topics naturally group together
- **Domain Separation**: Clear boundaries between different academic subjects
- **Embedding Quality**: How well the model captures semantic meaning
- **Clustering Validation**: Whether unsupervised clustering aligns with known labels

### Real-World Applications
- **Document Organization**: Automatically categorize and group similar documents
- **Content Discovery**: Find related articles, papers, or text segments
- **Search Enhancement**: Improve search results through semantic similarity
- **Data Exploration**: Understand large text corpora through visualization

## 🔧 Configuration

The project uses a `config.yaml` file to manage:
- **Model Selection**: BAAI/bge-base-en-v1.5 sentence encoder
- **Data Paths**: Location of pre-processed text chunks

## Subject Chunks Data
The `chunks.tar.gz` file should be separately downloaded from the data hub on the course portal and unzipped into a folder in your machine.
This folder will be updated as the value in the `config.yaml` under the `chunks_path` key under the `data` key.

## 📝 Notebook Structure

### Section 1: Introduction and Setup
- Project overview and objectives
- Library imports and configuration loading
- Environment setup (GPU/CPU detection)

### Section 2: Data Loading
- Loading pre-chunked text documents
- Dataset inspection and statistics
- Data filtering and preprocessing

### Section 3: Embedding Generation
- Sentence transformer model loading
- Batch processing for efficiency
- Vector generation and storage

### Section 4: Dimensionality Reduction
- t-SNE parameter configuration
- 2D coordinate generation
- Data preparation for visualization

### Section 5: Visualization and Clustering
- Labeled scatter plots
- K-means clustering implementation
- Cluster center visualization
- Pattern analysis and interpretation

## 🎓 Learning Outcomes

After completing this notebook, you will understand:

- **Text Representation**: How modern NLP models convert text to numbers
- **Dimensionality Reduction**: Why and how to reduce high-dimensional data
- **Clustering Algorithms**: How unsupervised learning groups similar data
- **Visualization Techniques**: Best practices for plotting high-dimensional data
- **Practical Applications**: Real-world use cases for text clustering

## Useful References

- [https://sbert.net/](https://sbert.net/)
- [https://huggingface.co/](https://huggingface.co/)
- [https://huggingface.co/spaces/mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard)

## 🚀 Next Steps

This notebook provides a foundation for:
- **Advanced Clustering**: Try different algorithms (DBSCAN, hierarchical clustering)
- **Interactive Visualization**: Use Plotly or Bokeh for dynamic plots
- **Custom Datasets**: Apply techniques to your own text data
- **Performance Optimization**: Implement GPU acceleration and parallel processing
- **Model Fine-tuning**: Adapt the embedding model for specific domains

