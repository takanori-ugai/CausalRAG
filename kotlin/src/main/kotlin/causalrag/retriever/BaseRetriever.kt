package causalrag.retriever

abstract class BaseRetriever(
    val name: String = "BaseRetriever",
) {
    open fun processQuery(query: String): String = query.trim()

    abstract fun retrieve(
        query: String,
        topK: Int = 5,
    ): List<Map<String, Any>>

    abstract fun indexCorpus(documents: List<String>): Boolean

    open fun saveIndex(path: String): Boolean = false

    open fun loadIndex(path: String): Boolean = false

    open fun getStats(): Map<String, Any> =
        mapOf(
            "name" to name,
            "type" to this::class.simpleName.orEmpty(),
        )
}

abstract class EmbeddingRetriever(
    name: String = "EmbeddingRetriever",
) : BaseRetriever(name) {
    abstract fun encodeQuery(query: String): DoubleArray

    abstract fun encodeDocuments(documents: List<String>): List<DoubleArray>
}

abstract class KeywordRetriever(
    name: String = "KeywordRetriever",
) : BaseRetriever(name) {
    open fun tokenize(text: String): List<String> = text.lowercase().split(WHITESPACE_REGEX)

    companion object {
        private val WHITESPACE_REGEX = Regex("\\s+")
    }
}
