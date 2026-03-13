package causalrag.retriever

/**
 * Base abstraction for retrievers that index a corpus and return scored passages.
 *
 * @property name Retriever name reported in diagnostics.
 */
abstract class BaseRetriever(
    val name: String = "BaseRetriever",
) {
    /**
     * Normalizes a query before retrieval.
     *
     * @param query Raw user query.
     * @return Normalized query string.
     */
    open fun processQuery(query: String): String = query.trim()

    /**
     * Retrieves the top matching passages for a query.
     *
     * @param query User query.
     * @param topK Maximum number of results to return.
     * @return Result maps that include passage and scoring metadata.
     */
    abstract fun retrieve(
        query: String,
        topK: Int = 5,
    ): List<Map<String, Any>>

    /**
     * Indexes a collection of documents.
     *
     * @param documents Documents to index.
     * @return `true` when indexing succeeds.
     */
    abstract fun indexCorpus(documents: List<String>): Boolean

    /**
     * Persists retriever state if the implementation supports it.
     *
     * @param path Destination path.
     * @return `true` when persistence succeeds.
     */
    open fun saveIndex(path: String): Boolean = false

    /**
     * Loads retriever state if the implementation supports it.
     *
     * @param path Source path.
     * @return `true` when loading succeeds.
     */
    open fun loadIndex(path: String): Boolean = false

    /**
     * Returns lightweight retriever statistics.
     *
     * @return Diagnostic key-value pairs.
     */
    open fun getStats(): Map<String, Any> =
        mapOf(
            "name" to name,
            "type" to this::class.simpleName.orEmpty(),
        )
}

/**
 * Base class for retrievers that rely on vector embeddings.
 */
abstract class EmbeddingRetriever(
    name: String = "EmbeddingRetriever",
) : BaseRetriever(name) {
    /**
     * Encodes a query for embedding-based retrieval.
     *
     * @param query User query.
     * @return Query embedding.
     */
    abstract fun encodeQuery(query: String): DoubleArray

    /**
     * Encodes documents for embedding-based retrieval.
     *
     * @param documents Documents to encode.
     * @return Embedding vectors aligned with [documents].
     */
    abstract fun encodeDocuments(documents: List<String>): List<DoubleArray>
}

/**
 * Base class for keyword-based retrievers.
 */
abstract class KeywordRetriever(
    name: String = "KeywordRetriever",
) : BaseRetriever(name) {
    /**
     * Tokenizes text for keyword indexing and matching.
     *
     * @param text Input text.
     * @return Token list.
     */
    open fun tokenize(text: String): List<String> = text.lowercase().split(WHITESPACE_REGEX)

    companion object {
        private val WHITESPACE_REGEX = Regex("\\s+")
    }
}
