package causalrag.retriever

import causalrag.utils.EmbeddingModel
import causalrag.utils.EmbeddingModelFactory
import causalrag.utils.SimpleHashEmbedding
import causalrag.utils.cosineSimilarity
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

@Suppress("TooGenericExceptionCaught")
class VectorStoreRetriever(
    embeddingModel: String = "all-MiniLM-L6-v2",
    private val backend: String = "memory",
    private val dimension: Int? = null,
    private val batchSize: Int = 32,
    private val cacheDir: String? = null,
    indexPath: String? = null,
    embeddingApiKey: String? = null,
    embeddingModelOverride: EmbeddingModel? = null,
) {
    private val encoder: EmbeddingModel? =
        embeddingModelOverride ?: run {
            if (!embeddingApiKey.isNullOrBlank()) {
                EmbeddingModelFactory.createOpenAi(embeddingModel, embeddingApiKey)
            } else {
                SimpleHashEmbedding(dimension ?: 384)
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

    fun indexCorpus(
        texts: List<String>,
        metadata: List<Map<String, Any>>? = null,
        ids: List<String>? = null,
        storeOriginal: Boolean = true,
    ) {
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

        if (storeOriginal) {
            passages.clear()
            passages.addAll(texts)
            this.metadata.clear()
            this.metadata.addAll(normalizedMetadata)
        }

        vectors.clear()
        for (i in texts.indices step batchSize) {
            val batch = texts.subList(i, minOf(i + batchSize, texts.size))
            val batchVectors = encoder.encodeAll(batch)
            vectors.addAll(batchVectors)
        }

        if (cacheDir != null) {
            cacheVectors(cacheDir, normalizedIds, normalizedMetadata)
        }

        logger.info { "Indexed ${texts.size} passages" }
    }

    fun search(
        query: String,
        topK: Int = 5,
        threshold: Double? = null,
        includeScores: Boolean = false,
    ): List<Any> {
        if (encoder == null) {
            logger.error { "Encoder not initialized, cannot search" }
            return emptyList()
        }
        if (vectors.isEmpty()) return emptyList()
        val qEmb = encoder.encode(query)
        val scores = mutableListOf<Pair<Int, Double>>()
        for (i in vectors.indices) {
            val sim = cosineSimilarity(qEmb, vectors[i])
            if (threshold == null || sim >= threshold) {
                scores.add(i to sim)
            }
        }
        val ranked = scores.sortedByDescending { it.second }.take(topK)
        return if (includeScores) {
            ranked.map { passages[it.first] to it.second }
        } else {
            ranked.map { passages[it.first] }
        }
    }

    fun searchWithMetadata(
        query: String,
        topK: Int = 5,
        threshold: Double? = null,
    ): List<Map<String, Any>> {
        val results = search(query, topK, threshold, includeScores = true)
        val output = mutableListOf<Map<String, Any>>()
        for ((idx, entry) in results.withIndex()) {
            @Suppress("UNCHECKED_CAST")
            val pair = entry as Pair<String, Double>
            val passage = pair.first
            val score = pair.second
            val passageIdx = passages.indexOf(passage)
            val meta = if (passageIdx in metadata.indices) metadata[passageIdx] else emptyMap()
            output.add(
                mapOf(
                    "passage" to passage,
                    "score" to score,
                    "metadata" to meta,
                    "rank" to (idx + 1),
                ),
            )
        }
        return output
    }

    private fun cacheVectors(
        dir: String,
        ids: List<String>,
        metadata: List<Map<String, Any>>,
    ) {
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
                }
            Files.writeString(Path.of(dir, "vectors.json"), json.encodeToString(JsonElement.serializer(), vectorsJson))
            Files.writeString(Path.of(dir, "metadata.json"), json.encodeToString(JsonElement.serializer(), metaJson))
            logger.info { "Cached vectors to $dir" }
        } catch (ex: IOException) {
            logger.error(ex) { "Error caching vectors" }
        } catch (ex: RuntimeException) {
            logger.error(ex) { "Error caching vectors" }
        }
    }

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
            vectors.clear()
            for (vecElement in vectorsJson) {
                if (vecElement is JsonArray) {
                    vectors.add(
                        vecElement
                            .mapNotNull { (it as? JsonPrimitive)?.content?.toDoubleOrNull() }
                            .toDoubleArray(),
                    )
                }
            }
            passages.clear()
            metadata.clear()
            val passagesJson = metaJson["passages"] as? JsonArray ?: JsonArray(emptyList())
            for (p in passagesJson) {
                passages.add((p as? JsonPrimitive)?.content.orEmpty())
            }
            val metaArray = metaJson["metadata"] as? JsonArray ?: JsonArray(emptyList())
            for (m in metaArray) {
                if (m is JsonObject) {
                    metadata.add(jsonObjectToMap(m))
                }
            }
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

    fun saveIndex(dir: String): Boolean =
        try {
            if (vectors.isEmpty() || passages.isEmpty()) {
                logger.warn { "No vectors to save" }
                false
            } else {
                val ids = passages.indices.map { it.toString() }
                cacheVectors(dir, ids, metadata)
                true
            }
        } catch (ex: IOException) {
            logger.error(ex) { "Error saving index" }
            false
        } catch (ex: RuntimeException) {
            logger.error(ex) { "Error saving index" }
            false
        }

    fun loadIndex(dir: String): Boolean = loadCached(dir)

    private fun mapToJson(map: Map<String, Any>): JsonObject =
        buildJsonObject {
            for ((k, v) in map) {
                when (v) {
                    is Number -> put(k, v.toString().toDouble())
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
}
