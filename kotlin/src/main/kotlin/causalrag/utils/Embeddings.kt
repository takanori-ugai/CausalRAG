package causalrag.utils

import io.github.oshai.kotlinlogging.KotlinLogging
import kotlin.math.sqrt

internal const val DEFAULT_LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
internal const val DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
private val logger = KotlinLogging.logger {}

/**
 * Minimal embedding model contract used across retrieval and graph components.
 */
interface EmbeddingModel {
    /**
     * Encodes a single string into a numeric embedding vector.
     *
     * @param text Input text.
     * @return Embedding vector for [text].
     */
    fun encode(text: String): DoubleArray

    /**
     * Encodes multiple strings in order.
     *
     * @param texts Input texts.
     * @return Embedding vectors aligned with [texts].
     */
    fun encodeAll(texts: List<String>): List<DoubleArray> = texts.map { encode(it) }
}

/**
 * Adapter that exposes a LangChain4j embedding model through [EmbeddingModel].
 */
class LangChain4jEmbeddingModel(
    private val delegate: dev.langchain4j.model.embedding.EmbeddingModel,
) : EmbeddingModel {
    override fun encode(text: String): DoubleArray {
        val embedding = delegate.embed(text).content()
        val vector = embedding.vector()
        return DoubleArray(vector.size) { idx -> vector[idx].toDouble() }
    }

    override fun encodeAll(texts: List<String>): List<DoubleArray> {
        val segments = texts.map(dev.langchain4j.data.segment.TextSegment::from)
        val embeddings = delegate.embedAll(segments).content()
        return embeddings.map { embedding ->
            val vector = embedding.vector()
            DoubleArray(vector.size) { idx -> vector[idx].toDouble() }
        }
    }
}

/**
 * Lightweight fallback embedding model based on token hashing.
 *
 * @param dimension Output vector size.
 */
class SimpleHashEmbedding(
    private val dimension: Int = 384,
) : EmbeddingModel {
    init {
        require(dimension > 0) { "dimension must be > 0" }
    }

    override fun encode(text: String): DoubleArray {
        val vector = DoubleArray(dimension)
        val tokens = tokenize(text)
        for (token in tokens) {
            val idx = (token.hashCode() and Int.MAX_VALUE) % dimension
            vector[idx] += 1.0
        }
        return l2Normalize(vector)
    }

    private fun tokenize(text: String): List<String> =
        Regex("""[\p{L}\p{M}\p{N}_]+""")
            .findAll(text.lowercase())
            .map { it.value }
            .toList()
}

/**
 * Computes cosine similarity between two embedding vectors.
 *
 * @param a First vector.
 * @param b Second vector.
 * @return Cosine similarity in the range `[-1, 1]`, or `0.0` when vectors are incompatible.
 */
fun cosineSimilarity(
    a: DoubleArray,
    b: DoubleArray,
): Double {
    if (a.isEmpty() || b.isEmpty() || a.size != b.size) return 0.0
    var dot = 0.0
    var normA = 0.0
    var normB = 0.0
    for (i in a.indices) {
        dot += a[i] * b[i]
        normA += a[i] * a[i]
        normB += b[i] * b[i]
    }
    if (normA == 0.0 || normB == 0.0) return 0.0
    return dot / (sqrt(normA) * sqrt(normB))
}

private fun l2Normalize(vector: DoubleArray): DoubleArray {
    var norm = 0.0
    for (v in vector) {
        norm += v * v
    }
    if (norm == 0.0) return vector
    val scale = 1.0 / sqrt(norm)
    return DoubleArray(vector.size) { index -> vector[index] * scale }
}

/**
 * Factory methods for constructing embedding backends used by the repository.
 */
object EmbeddingModelFactory {
    /**
     * Creates the default embedding model for the current environment.
     *
     * When an API key is available, this factory always returns an OpenAI-backed model. Passing the
     * local default name (`all-MiniLM-L6-v2`) in that case is treated as "use the default external
     * embedding model" and is resolved to `text-embedding-3-small`, because the local sentence-transformer
     * is not instantiated by this factory. When no API key is available, the factory falls back to
     * [SimpleHashEmbedding].
     *
     * @param modelName Embedding model name. The local default name is remapped to the OpenAI default
     * when [apiKey] is present.
     * @param apiKey Optional API key. When absent, falls back to [SimpleHashEmbedding].
     * @return Configured embedding model.
     */
    fun createDefault(
        modelName: String,
        apiKey: String? = System.getenv("OPENAI_API_KEY"),
    ): EmbeddingModel =
        if (!apiKey.isNullOrBlank()) {
            val resolvedModelName = resolveDefaultModelName(modelName, hasApiKey = true)
            if (resolvedModelName != modelName) {
                logger.info {
                    "Resolved embedding model '$modelName' to '$resolvedModelName' because createDefault() " +
                        "uses an OpenAI-backed encoder when an API key is configured."
                }
            }
            createOpenAi(resolvedModelName, apiKey)
        } else {
            SimpleHashEmbedding()
        }

    /**
     * Creates an OpenAI-backed embedding model.
     *
     * @param modelName Embedding model name.
     * @param apiKey OpenAI API key.
     * @return Configured embedding model.
     */
    fun createOpenAi(
        modelName: String,
        apiKey: String,
    ): EmbeddingModel {
        val openAi =
            dev.langchain4j.model.openai.OpenAiEmbeddingModel
                .builder()
                .apiKey(apiKey)
                .modelName(modelName)
                .build()
        return LangChain4jEmbeddingModel(openAi)
    }
}

internal fun resolveDefaultModelName(
    modelName: String,
    hasApiKey: Boolean,
): String =
    if (hasApiKey && modelName == DEFAULT_LOCAL_EMBEDDING_MODEL) {
        DEFAULT_OPENAI_EMBEDDING_MODEL
    } else {
        modelName
    }
