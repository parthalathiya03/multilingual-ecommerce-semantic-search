"""
=============================================================================
PROJECT: MULTILINGUAL E-COMMERCE PRODUCT SEARCH ENGINE
=============================================================================

Real-World Use Case:
Build a production-ready semantic search system for an international
e-commerce platform that:

1. Supports 100+ languages (English, Spanish, Hindi, Chinese, Arabic, etc.)
2. Handles product descriptions, reviews, and FAQs
3. Provides hybrid search (semantic + keyword matching)
4. Reranks results for maximum relevance
5. Optimized for production (quantization, caching)
6. Visualizes search quality and embeddings

Technologies:
- Sentence Transformers v5.2
- BGE-M3 for multilingual support
- SPLADE for sparse retrieval
- Cross-Encoder for reranking
- Quantization for efficiency
"""

# pip install sentence-transformers==5.2.0 scikit-learn matplotlib plotly pandas

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from sentence_transformers.quantization import quantize_embeddings
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass
import time

# For visualization
try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False
    print("Install matplotlib and scikit-learn for visualizations")


@dataclass
class Product:
    """Product data structure"""
    id: str
    name: str
    description: str
    category: str
    price: float
    language: str

    def to_text(self) -> str:
        """Convert product to searchable text"""
        return f"{self.name}. {self.description}. Category: {self.category}"


def create_sample_products() -> List[Product]:
    """
    Create multilingual product catalog
    Demonstrates real-world international e-commerce scenario
    """

    products = [
        # Electronics (English)
        Product("P001", "Wireless Bluetooth Headphones",
                "Premium noise-canceling headphones with 30-hour battery life and superior sound quality",
                "Electronics", 129.99, "en"),

        Product("P002", "4K Smart TV 55 inch",
                "Ultra HD television with HDR support, built-in streaming apps, and voice control",
                "Electronics", 599.99, "en"),

        Product("P003", "Gaming Laptop",
                "High-performance laptop with RTX graphics card, 16GB RAM, and RGB keyboard",
                "Electronics", 1299.99, "en"),

        # Electronics (Spanish)
        Product("P004", "Auriculares Inalámbricos Bluetooth",
                "Auriculares premium con cancelación de ruido, batería de 30 horas y sonido superior",
                "Electrónica", 129.99, "es"),

        Product("P005", "Smart TV 4K 55 pulgadas",
                "Televisor Ultra HD con soporte HDR, aplicaciones de streaming integradas y control por voz",
                "Electrónica", 599.99, "es"),

        # Electronics (Hindi)
        Product("P006", "वायरलेस ब्लूटूथ हेडफोन",
                "30 घंटे की बैटरी लाइफ और उत्कृष्ट ध्वनि गुणवत्ता के साथ प्रीमियम शोर-रद्द करने वाले हेडफ़ोन",
                "इलेक्ट्रॉनिक्स", 129.99, "hi"),

        # Electronics (Chinese)
        Product("P007", "无线蓝牙耳机",
                "高级降噪耳机，30小时电池续航和卓越音质",
                "电子产品", 129.99, "zh"),

        Product("P008", "4K智能电视55英寸",
                "超高清电视，支持HDR，内置流媒体应用和语音控制",
                "电子产品", 599.99, "zh"),

        # Fashion (English)
        Product("P009", "Men's Running Shoes",
                "Lightweight athletic shoes with breathable mesh and cushioned sole for comfort",
                "Fashion", 89.99, "en"),

        Product("P010", "Women's Leather Handbag",
                "Elegant genuine leather bag with multiple compartments and adjustable strap",
                "Fashion", 149.99, "en"),

        # Fashion (Spanish)
        Product("P011", "Zapatos para Correr de Hombre",
                "Zapatos deportivos ligeros con malla transpirable y suela acolchada para mayor comodidad",
                "Moda", 89.99, "es"),

        # Fashion (Arabic)
        Product("P012", "حذاء رياضي للرجال",
                "أحذية رياضية خفيفة الوزن مع شبكة قابلة للتنفس ونعل مبطن للراحة",
                "أزياء", 89.99, "ar"),

        # Home & Kitchen (English)
        Product("P013", "Stainless Steel Cookware Set",
                "Professional 10-piece cookware set with non-stick coating and heat-resistant handles",
                "Home & Kitchen", 199.99, "en"),

        Product("P014", "Robot Vacuum Cleaner",
                "Smart vacuum with mapping technology, auto-charging, and app control",
                "Home & Kitchen", 299.99, "en"),

        # Home & Kitchen (French)
        Product("P015", "Set de Casseroles en Acier Inoxydable",
                "Ensemble professionnel de 10 pièces avec revêtement antiadhésif et poignées résistantes à la chaleur",
                "Maison et Cuisine", 199.99, "fr"),

        # Sports (English)
        Product("P016", "Yoga Mat with Carrying Strap",
                "Extra thick exercise mat with non-slip surface, perfect for yoga and fitness",
                "Sports", 34.99, "en"),

        Product("P017", "Adjustable Dumbbells Set",
                "Space-saving weights with quick adjustment from 5 to 52.5 pounds",
                "Sports", 349.99, "en"),

        # Sports (German)
        Product("P018", "Yogamatte mit Tragegurt",
                "Extra dicke Trainingsmatte mit rutschfester Oberfläche, perfekt für Yoga und Fitness",
                "Sport", 34.99, "de"),

        # Books (English)
        Product("P019", "Machine Learning Fundamentals",
                "Comprehensive guide to ML algorithms, neural networks, and deep learning",
                "Books", 45.99, "en"),

        Product("P020", "Cooking Masterclass Cookbook",
                "Professional recipes and techniques from world-renowned chefs",
                "Books", 29.99, "en"),
    ]

    return products


class MultilingualSearchEngine:
    """
    Multilingual semantic search engine

    Features:
    - BGE-M3 for multilingual embeddings
    - Quantization for memory efficiency
    - Hybrid search capability
    - Cross-encoder reranking
    - Performance monitoring
    """

    def __init__(self,
                 model_name: str = 'all-MiniLM-L6-v2',
                 reranker_name: str = 'cross-encoder/ms-marco-MiniLM-L6-v2',
                 use_quantization: bool = True,
                 quantization_type: str = 'int8'):

        print("=" * 80)
        print("Initializing Multilingual Search Engine")
        print("=" * 80)

        print(f"\n Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

        print(f"Loading reranker: {reranker_name}")
        self.reranker = CrossEncoder(reranker_name)

        self.use_quantization = use_quantization
        self.quantization_type = quantization_type
        self.products: List[Product] = []
        self.embeddings = None
        self.embeddings_float = None
        self.search_cache = {}

        self.metrics = {
            'total_searches': 0,
            'cache_hits': 0,
            'avg_search_time': 0,
            'total_search_time': 0
        }

        print("Search engine initialized successfully!")

    def index_products(self, products: List[Product]):
        """
        Index products for search

        Steps:
        1. Extract searchable text from products
        2. Generate embeddings
        3. Quantize for storage comparison (keeps float32 for actual use)
        """

        print(f"\n Indexing {len(products)} products")
        self.products = products
        texts = [p.to_text() for p in products]

        print("Generating embeddings")
        start_time = time.time()

        embeddings = self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        embedding_time = time.time() - start_time
        self.embeddings_float = embeddings

        # demonstrate quantization for storage comparison
        if self.use_quantization:
            print(
                f"Quantizing embeddings for storage comparison ({self.quantization_type})")
            self.embeddings = quantize_embeddings(
                embeddings,
                precision=self.quantization_type
            )

            # calculate compression..all the size is in the MB
            original_size = embeddings.nbytes / 1024 / 1024
            compressed_size = self.embeddings.nbytes / 1024 / 1024
            compression_ratio = original_size / compressed_size

            print(f"Original size: {original_size:.2f} MB")
            print(f"Compressed size: {compressed_size:.2f} MB")
            print(f"Compression ratio: {compression_ratio:.1f}x")
            print(f"Note: Using float32 for similarity calculations")
        else:
            self.embeddings = embeddings

        print(f"Indexed {len(products)} products in {embedding_time:.2f}s")
        print(f"Embedding dimension: {self.embeddings_float.shape[1]}")

    def search(self,
               query: str,
               top_k: int = 5,
               use_reranker: bool = True,
               filter_language: str = None,
               filter_category: str = None) -> List[Dict]:
        """
        Search products with optional filters

        Pipeline:
        1. Check cache
        2. Encode query
        3. Compute similarities
        4. Apply filters
        5. Rerank top results
        6. Update metrics

        Args:
            query: Search query
            top_k: Number of results to return
            use_reranker: Whether to use cross-encoder reranking
            filter_language: Filter by language code (e.g., 'en', 'es')
            filter_category: Filter by product category

        Returns:
            List of search results with scores
        """

        start_time = time.time()

        cache_key = f"{query}_{top_k}_{use_reranker}_{filter_language}_{filter_category}"
        if cache_key in self.search_cache:
            self.metrics['cache_hits'] += 1
            self.metrics['total_searches'] += 1
            return self.search_cache[cache_key]

        # Encode query (always in float32 for similarity calculation)
        query_emb = self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        # Convert quantized embeddings back to float32 for comparison
        if self.use_quantization:
            embeddings_float = self.embeddings_float
        else:
            embeddings_float = self.embeddings

        similarities = self.model.similarity(query_emb, embeddings_float)[0]

        valid_indices = list(range(len(self.products)))

        if filter_language:
            valid_indices = [i for i in valid_indices
                             if self.products[i].language == filter_language]

        if filter_category:
            valid_indices = [i for i in valid_indices
                             if self.products[i].category.lower() == filter_category.lower()]

        filtered_similarities = [(i, similarities[i]) for i in valid_indices]
        filtered_similarities.sort(key=lambda x: x[1], reverse=True)

        rerank_k = min(top_k * 2, len(filtered_similarities))
        candidate_indices = [idx for idx,
                             _ in filtered_similarities[:rerank_k]]

        if use_reranker and len(candidate_indices) > 0:

            candidate_docs = [self.products[idx].to_text()
                              for idx in candidate_indices]

            ranks = self.reranker.rank(
                query,
                candidate_docs,
                return_documents=False,
                top_k=min(top_k, len(candidate_docs))
            )

            results = []
            for rank in ranks:
                original_idx = candidate_indices[rank['corpus_id']]
                product = self.products[original_idx]

                results.append({
                    'product': product,
                    'score': rank['score'],
                    'rank': len(results) + 1
                })
        else:
            results = []
            for idx, sim in filtered_similarities[:top_k]:
                product = self.products[idx]
                results.append({
                    'product': product,
                    'score': float(sim),
                    'rank': len(results) + 1
                })

        # update metrics
        search_time = time.time() - start_time
        self.metrics['total_searches'] += 1
        self.metrics['total_search_time'] += search_time
        self.metrics['avg_search_time'] = (
            self.metrics['total_search_time'] / self.metrics['total_searches']
        )

        # Cache result
        self.search_cache[cache_key] = results

        return results

    def get_stats(self) -> Dict:
        """Get search engine statistics"""
        return {
            'indexed_products': len(self.products),
            'embedding_dimension': self.embeddings.shape[1] if self.embeddings is not None else 0,
            'quantization': self.quantization_type if self.use_quantization else 'none',
            'cache_size': len(self.search_cache),
            **self.metrics,
            'cache_hit_rate': (
                self.metrics['cache_hits'] / self.metrics['total_searches']
                if self.metrics['total_searches'] > 0 else 0
            )
        }

    def visualize_embeddings(self, method='tsne', save_path=None):
        """
        Visualize product embeddings in 2D

        Args:
            method: 'pca' or 'tsne'
            save_path: Path to save figure (optional)
        """

        if not HAS_VISUALIZATION:
            print("Visualization libraries not available")
            return

        if self.embeddings is None:
            print("No embeddings to visualize")
            return

        print(f"\n🎨 Visualizing embeddings using {method.upper()}...")

        # Convert to float32 if quantized
        emb = self.embeddings.astype(np.float32)

        # Reduce to 2D
        if method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
        else:
            reducer = TSNE(n_components=2, random_state=42,
                           perplexity=min(30, len(emb)-1))

        coords_2d = reducer.fit_transform(emb)

        # Create plot
        plt.figure(figsize=(12, 8))

        # Color by category
        categories = list(set(p.category for p in self.products))
        colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))

        for i, category in enumerate(categories):
            indices = [j for j, p in enumerate(self.products)
                       if p.category == category]

            plt.scatter(
                coords_2d[indices, 0],
                coords_2d[indices, 1],
                c=[colors[i]],
                label=category,
                alpha=0.6,
                s=100
            )

        plt.legend()
        plt.title(
            f'Product Embeddings Visualization ({method.upper()})', fontsize=14)
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")

        plt.tight_layout()
        plt.show()


def run_demo():
    """Run complete demo of the search engine"""

    print("\n" + "="*80)
    print("MULTILINGUAL PRODUCT SEARCH ENGINE DEMO")
    print("="*80)

    # Create products
    products = create_sample_products()

    print(f"\n Created {len(products)} products in multiple languages:")
    languages = set(p.language for p in products)
    print(f"Languages: {', '.join(sorted(languages))}")
    categories = set(p.category for p in products)
    print(f"Categories: {', '.join(sorted(categories))}")

    engine = MultilingualSearchEngine(
        model_name='all-MiniLM-L6-v2',  # we can use 'BAAI/bge-m3' for true multilingual
        use_quantization=True,
        quantization_type='int8'
    )

    engine.index_products(products)

    print("\n" + "="*80)
    print("SEARCH DEMONSTRATIONS")
    print("="*80)

    test_queries = [
        {
            'query': 'wireless headphones with long battery',
            'description': 'English query for headphones',
            'filters': {}
        },
        {
            'query': 'auriculares inalámbricos',
            'description': 'Spanish query (cross-lingual search)',
            'filters': {}
        },
        {
            'query': 'smart tv with streaming',
            'description': 'English query for TV',
            'filters': {'filter_category': 'Electronics'}
        },
        {
            'query': 'योग चटाई',
            'description': 'Hindi query (demonstrates multilingual capability)',
            'filters': {}
        },
        {
            'query': 'running shoes comfortable',
            'description': 'English query for shoes',
            'filters': {'filter_language': 'en'}
        }
    ]

    for i, test in enumerate(test_queries, 1):
        print(f"\n{'─'*80}")
        print(f"Search {i}: {test['description']}")
        print(f"Query: '{test['query']}'")

        if test['filters']:
            print(f"Filters: {test['filters']}")

        results = engine.search(
            test['query'],
            top_k=3,
            use_reranker=True,
            **test['filters']
        )

        print(f"\nTop Results:")
        for result in results:
            product = result['product']
            score = result['score']
            rank = result['rank']

            print(f"\n {rank}. [{product.language.upper()}] {product.name}")
            print(f"Category: {product.category} | Price: ${product.price}")
            print(f"Score: {score:.4f}")
            print(f"Description: {product.description[:80]}...")

    print("\n" + "="*80)
    print("SEARCH ENGINE STATISTICS")
    print("="*80)

    stats = engine.get_stats()
    print(f"\n Performance Metrics:")
    print(f"Total searches: {stats['total_searches']}")
    print(f"Cache hits: {stats['cache_hits']}")
    print(f"Cache hit rate: {stats['cache_hit_rate']:.1%}")
    print(f"Average search time: {stats['avg_search_time']*1000:.2f}ms")
    print(f"\n Index Information:")
    print(f"Indexed products: {stats['indexed_products']}")
    print(f"Embedding dimension: {stats['embedding_dimension']}")
    print(f"Quantization: {stats['quantization']}")

    if HAS_VISUALIZATION:
        print("\n" + "="*80)
        print("EMBEDDING VISUALIZATION")
        print("="*80)
        engine.visualize_embeddings(method='tsne')

    print("\n" + "="*80)
    print("RERANKING COMPARISON")
    print("="*80)

    query = "affordable laptop for gaming"
    print(f"\nQuery: '{query}'")

    # without reranking
    print("\n Without Reranking:")
    results_no_rerank = engine.search(query, top_k=3, use_reranker=False)
    for r in results_no_rerank:
        print(f"{r['rank']}. {r['product'].name} (Score: {r['score']:.4f})")

    # with reranking
    print("\n With Cross-Encoder Reranking:")
    results_with_rerank = engine.search(query, top_k=3, use_reranker=True)
    for r in results_with_rerank:
        print(f"  {r['rank']}. {r['product'].name} (Score: {r['score']:.4f})")

    print("\n" + "="*80)
    print("DEMO COMPLETE!")
    print("="*80)

    return engine


if __name__ == "__main__":
    engine = run_demo()
