package causalrag.retriever

import causalrag.utils.DEFAULT_LOCAL_EMBEDDING_MODEL
import causalrag.utils.DEFAULT_OPENAI_EMBEDDING_MODEL
import causalrag.utils.EmbeddingModel
import causalrag.utils.EmbeddingModelFactory
import causalrag.utils.SimpleHashEmbedding
import causalrag.utils.cosineSimilarity
import causalrag.utils.resolveDefaultModelName
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import java.io.IOException
import java.nio.file.Files
import java.nio.file.Path

private val logger = KotlinLogging.logger {}
private const val DEFAULT_HASH_DIMENSION = 384
private val OPENAI_EMBEDDING_DIMENSIONS =
    mapOf(
        "text-embedding-3-small" to 1536,
        "text-embedding-3-large" to 3072,
        "text-embedding-ada-002" to 1536,
    )

/**
 * In-memory vector retriever with optional JSON-based caching.
 *
 * @param embeddingModel Embedding model name used when creating an encoder internally.
 * @param backend Vector store backend label; the current implementation stores vectors in memory.
 * @param dimension Optional embedding dimension for fallback hash-based embeddings.
 * @param batchSize Number of texts encoded per batch during indexing.
 * @param cacheDir Optional directory used to persist cached vectors and metadata.
 * @param indexPath Optional path to a previously cached index loaded during initialization.
 * @param embeddingApiKey Optional API key used when creating an external embedding model.
 * @param embeddingModelOverride Optional pre-configured [EmbeddingModel] used instead of creating one.
 */
@Suppress("TooGenericExceptionCaught")
class VectorStoreRetriever(
    embeddingModel: String = DEFAULT_LOCAL_EMBEDDING_MODEL,
    private val backend: String = "memory",
    private val dimension: Int? = null,
    private val batchSize: Int = 32,
    private val cacheDir: String? = null,
    indexPath: String? = null,
    embeddingApiKey: String? = null,
    embeddingModelOverride: EmbeddingModel? = null,
) {
    private val resolvedEmbeddingModel =
        resolveEmbeddingModelName(
            configuredModel = embeddingModel,
            embeddingApiKey = embeddingApiKey,
            embeddingModelOverride = embeddingModelOverride,
        )
    private val expectedVectorDimension =
        resolveExpectedVectorDimension(
            modelName = resolvedEmbeddingModel,
            embeddingApiKey = embeddingApiKey,
            embeddingModelOverride = embeddingModelOverride,
        )
    private val encoder: EmbeddingModel? =
        embeddingModelOverride ?: run {
            if (!embeddingApiKey.isNullOrBlank()) {
                EmbeddingModelFactory.createOpenAi(resolvedEmbeddingModel, embeddingApiKey)
            } else {
                SimpleHashEmbedding(expectedVectorDimension ?: DEFAULT_HASH_DIMENSION)
            }
        }
    private val vectors = mutableListOf<DoubleArray>()
    private val passages = mutableListOf<String>()
    private val metadata = mutableListOf<Map<String, Any>>()

    init {
        logger.info { "Vector store backend: $backend" }
        if (indexPath != null) {
            loadCached(indexPath)
        }
    }

    /**
     * Indexes a corpus of texts into the vector store.
     *
     * @param texts Text passages to index.
     * @param metadata Optional metadata aligned with [texts].
     * @param ids Optional identifiers aligned with [texts].
     * @param storeOriginal Whether to retain the original passages and metadata. Must be `true`
     * because the current search API returns passages from the in-memory index.
     */
    fun indexCorpus(
        texts: List<String>,
        metadata: List<Map<String, Any>>? = null,
        ids: List<String>? = null,
        storeOriginal: Boolean = true,
    ) {
        require(storeOriginal) {
            "storeOriginal=false is not supported because the current search API requires stored passages."
        }
        if (texts.isEmpty()) {
            logger.warn { "Empty corpus provided for indexing" }
            return
        }
        if (encoder == null) {
            logger.error { "Encoder not initialized, cannot index corpus" }
            return
        }
        val normalizedMetadata =
            when {
                metadata == null -> {
                    texts.mapIndexed { idx, _ -> mapOf("id" to idx.toString(), "position" to idx) }
                }

                metadata.size != texts.size -> {
                    logger.warn { "Metadata length (${metadata.size}) doesn't match texts (${texts.size})" }
                    texts.mapIndexed { idx, _ -> mapOf("id" to idx.toString(), "position" to idx) }
                }

                else -> {
                    metadata
                }
            }
        val normalizedIds =
            when {
                ids == null -> {
                    texts.indices.map { it.toString() }
                }

                ids.size != texts.size -> {
                    logger.warn { "IDs length (${ids.size}) doesn't match texts (${texts.size})" }
                    texts.indices.map { it.toString() }
                }

                else -> {
                    ids
                }
            }

        val newPassages = texts.toMutableList()
        val newMetadata = normalizedMetadata.toMutableList()
        val newVectors = mutableListOf<DoubleArray>()
        for (i in texts.indices step batchSize) {
            val batch = texts.subList(i, minOf(i + batchSize, texts.size))
            val batchVectors = encoder.encodeAll(batch)
            newVectors.addAll(batchVectors)
        }
        passages.clear()
        passages.addAll(newPassages)
        this.metadata.clear()
        this.metadata.addAll(newMetadata)
        vectors.clear()
        vectors.addAll(newVectors)

        if (cacheDir != null) {
            cacheVectors(cacheDir, normalizedIds, normalizedMetadata)
        }

        logger.info { "Indexed ${texts.size} passages" }
    }

    /**
     * Searches the indexed corpus and returns matching passages only.
     *
     * @param query User query.
     * @param topK Maximum number of matches to return.
     * @param threshold Optional minimum cosine similarity.
     * @return Matching passages ordered by score.
     */
    fun search(
        query: String,
        topK: Int = 5,
        threshold: Double? = null,
    ): List<String> = searchIndexed(query, topK, threshold).map { it.passage }

    /**
     * Searches the indexed corpus and returns passage-score pairs.
     *
     * @param query User query.
     * @param topK Maximum number of matches to return.
     * @param threshold Optional minimum cosine similarity.
     * @return Matching passages with similarity scores.
     */
    fun searchWithScores(
        query: String,
        topK: Int = 5,
        threshold: Double? = null,
    ): List<Pair<String, Double>> = searchIndexed(query, topK, threshold).map { it.passage to it.score }

    private data class SearchHit(
        val index: Int,
        val passage: String,
        val score: Double,
    )

    private fun searchIndexed(
        query: String,
        topK: Int,
        threshold: Double?,
    ): List<SearchHit> {
        require(topK >= 0) { "topK must be >= 0" }
        if (encoder == null) {
            logger.error { "Encoder not initialized, cannot search" }
            return emptyList()
        }
        if (vectors.isEmpty()) return emptyList()
        val qEmb = encoder.encode(query)
        if (passages.size != vectors.size) {
            logger.error { "Indexed data is inconsistent: vectors=${vectors.size}, passages=${passages.size}" }
            return emptyList()
        }
        val vectorDimension = vectors.firstOrNull()?.size ?: return emptyList()
        if (qEmb.size != vectorDimension) {
            logger.error {
                "Query embedding dimension (${qEmb.size}) does not match indexed vectors ($vectorDimension)"
            }
            return emptyList()
        }
        val scores = mutableListOf<Pair<Int, Double>>()
        for (i in vectors.indices) {
            val sim = cosineSimilarity(qEmb, vectors[i])
            if (threshold == null || sim >= threshold) {
                scores.add(i to sim)
            }
        }
        val ranked = scores.sortedByDescending { it.second }.take(topK)
        return ranked.map { (idx, score) ->
            SearchHit(idx, passages[idx], score)
        }
    }

    /**
     * Searches the indexed corpus and returns passages with metadata.
     *
     * @param query User query.
     * @param topK Maximum number of matches to return.
     * @param threshold Optional minimum cosine similarity.
     * @return Result maps including passage text, metadata, score, and rank.
     */
    fun searchWithMetadata(
        query: String,
        topK: Int = 5,
        threshold: Double? = null,
    ): List<Map<String, Any>> {
        val results = searchIndexed(query, topK, threshold)
        val output = mutableListOf<Map<String, Any>>()
        for ((rank, entry) in results.withIndex()) {
            val meta = if (entry.index in metadata.indices) metadata[entry.index] else emptyMap()
            output.add(
                mapOf(
                    "passage" to entry.passage,
                    "score" to entry.score,
                    "metadata" to meta,
                    "rank" to (rank + 1),
                ),
            )
        }
        return output
    }

    private fun cacheVectors(
        dir: String,
        ids: List<String>,
        metadata: List<Map<String, Any>>,
    ): Boolean =
        try {
            Files.createDirectories(Path.of(dir))
            val json = Json { prettyPrint = true }
            val vectorsJson =
                buildJsonArray {
                    vectors.forEach { vec ->
                        add(buildJsonArray { vec.forEach { add(JsonPrimitive(it)) } })
                    }
                }
            val metaJson =
                buildJsonObject {
                    put("ids", buildJsonArray { ids.forEach { add(JsonPrimitive(it)) } })
                    put("metadata", buildJsonArray { metadata.forEach { add(mapToJson(it)) } })
                    put("passages", buildJsonArray { passages.forEach { add(JsonPrimitive(it)) } })
                    put("backend", JsonPrimitive(backend))
                    put("embedding_model", JsonPrimitive(resolvedEmbeddingModel))
                    put("vector_dimension", JsonPrimitive(vectors.firstOrNull()?.size ?: 0))
                }
            Files.writeString(Path.of(dir, "vectors.json"), json.encodeToString(JsonElement.serializer(), vectorsJson))
            Files.writeString(Path.of(dir, "metadata.json"), json.encodeToString(JsonElement.serializer(), metaJson))
            logger.info { "Cached vectors to $dir" }
            true
        } catch (ex: IOException) {
            logger.error(ex) { "Error caching vectors" }
            false
        } catch (ex: RuntimeException) {
            logger.error(ex) { "Error caching vectors" }
            false
        }

    /**
     * Loads cached vectors and metadata from disk.
     *
     * @param cacheDir Directory containing `vectors.json` and `metadata.json`.
     * @return `true` when the cache is loaded successfully.
     */
    fun loadCached(cacheDir: String): Boolean {
        val vectorsPath = Path.of(cacheDir, "vectors.json")
        val metaPath = Path.of(cacheDir, "metadata.json")
        if (!Files.exists(vectorsPath) || !Files.exists(metaPath)) {
            logger.error { "Cache files not found in $cacheDir" }
            return false
        }
        return try {
            val json = Json { ignoreUnknownKeys = true }
            val vectorsJson = json.parseToJsonElement(Files.readString(vectorsPath)) as? JsonArray
            val metaJson = json.parseToJsonElement(Files.readString(metaPath)) as? JsonObject
            if (vectorsJson == null || metaJson == null) return false
            val loadedVectors = mutableListOf<DoubleArray>()
            val loadedPassages = mutableListOf<String>()
            val loadedMetadata = mutableListOf<Map<String, Any>>()
            for (vecElement in vectorsJson) {
                if (vecElement is JsonArray) {
                    loadedVectors.add(
                        vecElement
                            .mapNotNull { (it as? JsonPrimitive)?.content?.toDoubleOrNull() }
                            .toDoubleArray(),
                    )
                }
            }
            val passagesJson = metaJson["passages"] as? JsonArray ?: JsonArray(emptyList())
            for (p in passagesJson) {
                loadedPassages.add((p as? JsonPrimitive)?.content.orEmpty())
            }
            val idsJson = metaJson["ids"] as? JsonArray
            if (idsJson != null && idsJson.size != loadedPassages.size) {
                logger.error { "Cached index is inconsistent: ids=${idsJson.size}, passages=${loadedPassages.size}" }
                return false
            }
            val cachedEmbeddingModel = (metaJson["embedding_model"] as? JsonPrimitive)?.content
            if (cachedEmbeddingModel != null && cachedEmbeddingModel != resolvedEmbeddingModel) {
                logger.error {
                    "Cached index embedding model ($cachedEmbeddingModel) does not match retriever configuration ($resolvedEmbeddingModel)"
                }
                return false
            }
            val metadataElement = metaJson["metadata"]
            when (metadataElement) {
                null -> {
                    repeat(loadedPassages.size) { loadedMetadata.add(emptyMap()) }
                }

                is JsonArray -> {
                    if (metadataElement.size != loadedPassages.size) {
                        logger.error {
                            "Cached index is inconsistent: metadata=${metadataElement.size}, passages=${loadedPassages.size}"
                        }
                        return false
                    }
                    for (m in metadataElement) {
                        if (m !is JsonObject) {
                            logger.error { "Cached index is inconsistent: metadata entries must be JSON objects" }
                            return false
                        }
                        loadedMetadata.add(jsonObjectToMap(m))
                    }
                }

                else -> {
                    logger.error { "Cached index is inconsistent: metadata field must be an array" }
                    return false
                }
            }
            val vectorSizes = loadedVectors.map { it.size }.toSet()
            if (vectorSizes.size > 1) {
                logger.error { "Cached index is inconsistent: vectors contain multiple dimensions $vectorSizes" }
                return false
            }
            val actualVectorDimension = vectorSizes.firstOrNull() ?: 0
            val cachedVectorDimension = (metaJson["vector_dimension"] as? JsonPrimitive)?.content?.toIntOrNull()
            if (cachedVectorDimension != null && cachedVectorDimension != actualVectorDimension) {
                logger.error {
                    "Cached index is inconsistent: metadata dimension=$cachedVectorDimension, actual=$actualVectorDimension"
                }
                return false
            }
            if (loadedPassages.size != loadedVectors.size) {
                logger.error { "Cached index is inconsistent: vectors=${loadedVectors.size}, passages=${loadedPassages.size}" }
                return false
            }
            if (expectedVectorDimension != null && actualVectorDimension != expectedVectorDimension) {
                logger.error {
                    "Cached index dimension ($actualVectorDimension) does not match retriever expectation ($expectedVectorDimension)"
                }
                return false
            }
            clearIndex()
            vectors.addAll(loadedVectors)
            passages.addAll(loadedPassages)
            metadata.addAll(loadedMetadata)
            logger.info { "Loaded ${vectors.size} vectors from cache" }
            true
        } catch (ex: IOException) {
            logger.error(ex) { "Error loading cached vectors" }
            false
        } catch (ex: RuntimeException) {
            logger.error(ex) { "Error loading cached vectors" }
            false
        }
    }

    /**
     * Saves the current vector index to disk.
     *
     * @param dir Destination directory.
     * @return `true` when the index is saved successfully.
     */
    fun saveIndex(dir: String): Boolean =
        try {
            if (vectors.isEmpty() || passages.isEmpty()) {
                logger.warn { "No vectors to save" }
                false
            } else {
                val ids = passages.indices.map { it.toString() }
                cacheVectors(dir, ids, metadata)
            }
        } catch (ex: IOException) {
            logger.error(ex) { "Error saving index" }
            false
        } catch (ex: RuntimeException) {
            logger.error(ex) { "Error saving index" }
            false
        }

    /**
     * Loads a previously saved vector index from disk.
     *
     * @param dir Source directory.
     * @return `true` when the index is loaded successfully.
     */
    fun loadIndex(dir: String): Boolean = loadCached(dir)

    /**
     * Returns the passages currently stored in the vector index.
     *
     * @return Indexed passages in storage order.
     */
    fun getPassages(): List<String> = passages.toList()

    private fun clearIndex() {
        vectors.clear()
        passages.clear()
        metadata.clear()
    }

    private fun mapToJson(map: Map<String, Any>): JsonObject =
        buildJsonObject {
            for ((k, v) in map) {
                when (v) {
                    is Int -> put(k, v)
                    is Long -> put(k, v)
                    is Number -> put(k, v.toDouble())
                    is Boolean -> put(k, v)
                    else -> put(k, v.toString())
                }
            }
        }

    private fun jsonObjectToMap(obj: JsonObject): Map<String, Any> {
        val result = mutableMapOf<String, Any>()
        for ((k, v) in obj) {
            when (v) {
                is JsonPrimitive -> {
                    result[k] = v.content.toBooleanStrictOrNull() ?: v.content.toDoubleOrNull() ?: v.content
                }

                else -> {
                    result[k] = v.toString()
                }
            }
        }
        return result
    }

    private fun resolveEmbeddingModelName(
        configuredModel: String,
        embeddingApiKey: String?,
        embeddingModelOverride: EmbeddingModel?,
    ): String {
        if (embeddingModelOverride != null || embeddingApiKey.isNullOrBlank()) {
            return configuredModel
        }
        val resolvedModel = resolveDefaultModelName(configuredModel, hasApiKey = true)
        if (resolvedModel != configuredModel) {
            logger.info {
                "Using $DEFAULT_OPENAI_EMBEDDING_MODEL for OpenAI embeddings instead of local default $DEFAULT_LOCAL_EMBEDDING_MODEL"
            }
        }
        return resolvedModel
    }

    private fun resolveExpectedVectorDimension(
        modelName: String,
        embeddingApiKey: String?,
        embeddingModelOverride: EmbeddingModel?,
    ): Int? =
        when {
            embeddingModelOverride != null -> dimension
            embeddingApiKey.isNullOrBlank() -> dimension ?: DEFAULT_HASH_DIMENSION
            else -> OPENAI_EMBEDDING_DIMENSIONS[modelName]
        }
}
