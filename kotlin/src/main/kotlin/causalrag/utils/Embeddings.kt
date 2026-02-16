package causalrag.utils

import kotlin.math.abs
import kotlin.math.sqrt

interface EmbeddingModel {
    fun encode(text: String): DoubleArray

    fun encodeAll(texts: List<String>): List<DoubleArray> = texts.map { encode(it) }
}

class LangChain4jEmbeddingModel(
    private val delegate: dev.langchain4j.model.embedding.EmbeddingModel,
) : EmbeddingModel {
    override fun encode(text: String): DoubleArray {
        val embedding = delegate.embed(text).content()
        val vector = embedding.vector()
        return DoubleArray(vector.size) { idx -> vector[idx].toDouble() }
    }

    override fun encodeAll(texts: List<String>): List<DoubleArray> = texts.map { encode(it) }
}

class SimpleHashEmbedding(
    private val dimension: Int = 384,
) : EmbeddingModel {
    override fun encode(text: String): DoubleArray {
        val vector = DoubleArray(dimension)
        val tokens = tokenize(text)
        for (token in tokens) {
            val idx = abs(token.hashCode()) % dimension
            vector[idx] += 1.0
        }
        return l2Normalize(vector)
    }

    private fun tokenize(text: String): List<String> =
        Regex("[A-Za-z0-9_]+")
            .findAll(text.lowercase())
            .map { it.value }
            .toList()
}

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
    for (i in vector.indices) {
        vector[i] *= scale
    }
    return vector
}

object EmbeddingModelFactory {
    fun createDefault(
        modelName: String,
        apiKey: String? = System.getenv("OPENAI_API_KEY"),
    ): EmbeddingModel =
        if (!apiKey.isNullOrBlank()) {
            createOpenAi(modelName, apiKey)
        } else {
            SimpleHashEmbedding()
        }

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
