package causalrag

import causalrag.retriever.VectorStoreRetriever
import causalrag.utils.EmbeddingModel
import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Tests vector index persistence and cache integrity checks.
 */
class TestVectorStoreRetriever {
    private lateinit var tempDir: Path

    /**
     * Creates a temporary directory for each test.
     */
    @BeforeTest
    fun setUp() {
        tempDir = Files.createTempDirectory("causalrag-vectors")
    }

    /**
     * Removes temporary test files and directories.
     */
    @AfterTest
    fun tearDown() {
        tempDir.toFile().deleteRecursively()
    }

    /**
     * Verifies that disabling passage retention is rejected until the API supports it.
     */
    @Test
    fun testIndexCorpusRejectsStoreOriginalFalse() {
        val retriever = VectorStoreRetriever(dimension = 64)

        val error =
            assertFailsWith<IllegalArgumentException> {
                retriever.indexCorpus(listOf("alpha signal"), storeOriginal = false)
            }

        assertTrue(error.message!!.contains("storeOriginal=false"))
        assertTrue(retriever.getPassages().isEmpty())
    }

    /**
     * Verifies that saveIndex reports failure when the target cannot be created as a directory.
     */
    @Test
    fun testSaveIndexFailsWhenCacheWriteFails() {
        val retriever = VectorStoreRetriever(dimension = 64)
        retriever.indexCorpus(listOf("cached document"))
        val filePath = Files.createTempFile(tempDir, "vector-cache", ".tmp")

        val saved = retriever.saveIndex(filePath.toString())

        assertFalse(saved)
    }

    /**
     * Verifies that malformed caches with mismatched passage and vector counts are rejected.
     */
    @Test
    fun testLoadIndexRejectsMismatchedPassageCount() {
        val cacheDir = tempDir.resolve("bad-count-cache")
        Files.createDirectories(cacheDir)
        Files.writeString(cacheDir.resolve("vectors.json"), "[[1.0, 0.0, 0.0]]")
        Files.writeString(
            cacheDir.resolve("metadata.json"),
            """
            {
              "passages": [],
              "metadata": [],
              "vector_dimension": 3
            }
            """.trimIndent(),
        )

        val retriever = VectorStoreRetriever(dimension = 3)
        val loaded = retriever.loadIndex(cacheDir.toString())

        assertFalse(loaded)
        assertTrue(retriever.getPassages().isEmpty())
    }

    /**
     * Verifies that malformed caches with misaligned metadata arrays are rejected.
     */
    @Test
    fun testLoadIndexRejectsMismatchedMetadataCount() {
        val cacheDir = tempDir.resolve("bad-metadata-cache")
        Files.createDirectories(cacheDir)
        Files.writeString(cacheDir.resolve("vectors.json"), "[[1.0, 0.0, 0.0]]")
        Files.writeString(
            cacheDir.resolve("metadata.json"),
            """
            {
              "passages": ["document"],
              "metadata": [],
              "vector_dimension": 3
            }
            """.trimIndent(),
        )

        val retriever = VectorStoreRetriever(dimension = 3)
        val loaded = retriever.loadIndex(cacheDir.toString())

        assertFalse(loaded)
        assertTrue(retriever.getPassages().isEmpty())
    }

    /**
     * Verifies that a failed cache load does not destroy the existing in-memory index.
     */
    @Test
    fun testFailedLoadPreservesExistingIndex() {
        val cacheDir = tempDir.resolve("bad-reload-cache")
        Files.createDirectories(cacheDir)
        Files.writeString(cacheDir.resolve("vectors.json"), "[[1.0, 0.0, 0.0]]")
        Files.writeString(
            cacheDir.resolve("metadata.json"),
            """
            {
              "passages": [],
              "metadata": [],
              "vector_dimension": 3
            }
            """.trimIndent(),
        )

        val retriever = VectorStoreRetriever(dimension = 64)
        retriever.indexCorpus(listOf("kept document"))

        val loaded = retriever.loadIndex(cacheDir.toString())

        assertFalse(loaded)
        assertEquals(listOf("kept document"), retriever.getPassages())
        assertEquals(listOf("kept document"), retriever.search("kept", topK = 1))
    }

    /**
     * Verifies that a failed reindex does not destroy the existing in-memory index.
     */
    @Test
    fun testFailedIndexCorpusPreservesExistingIndex() {
        val failingEmbedding =
            object : EmbeddingModel {
                override fun encode(text: String): DoubleArray = doubleArrayOf(text.length.toDouble())

                override fun encodeAll(texts: List<String>): List<DoubleArray> {
                    if ("explode" in texts) {
                        error("synthetic embedding failure")
                    }
                    return texts.map { encode(it) }
                }
            }
        val failingRetriever = VectorStoreRetriever(embeddingModelOverride = failingEmbedding, batchSize = 1)
        failingRetriever.indexCorpus(listOf("kept document"))

        val error =
            assertFailsWith<IllegalStateException> {
                failingRetriever.indexCorpus(listOf("safe", "explode"))
            }

        assertTrue(error.message!!.contains("synthetic embedding failure"))
        assertEquals(listOf("kept document"), failingRetriever.getPassages())
        assertEquals(listOf("kept document"), failingRetriever.search("kept", topK = 1))
    }

    /**
     * Verifies that negative topK values are rejected consistently through the public search API.
     */
    @Test
    fun testSearchRejectsNegativeTopK() {
        val retriever = VectorStoreRetriever(dimension = 64)
        retriever.indexCorpus(listOf("kept document"))

        val error =
            assertFailsWith<IllegalArgumentException> {
                retriever.search("kept", topK = -1)
            }

        assertTrue(error.message!!.contains("topK must be >= 0"))
    }

    /**
     * Verifies that caches built with a different vector dimension are rejected.
     */
    @Test
    fun testLoadIndexRejectsDimensionMismatch() {
        val cacheDir = tempDir.resolve("dimension-cache")
        val writer = VectorStoreRetriever(dimension = 64)
        writer.indexCorpus(listOf("dimensioned document"))
        assertTrue(writer.saveIndex(cacheDir.toString()))

        val reader = VectorStoreRetriever(dimension = 32)
        val loaded = reader.loadIndex(cacheDir.toString())

        assertFalse(loaded)
        assertTrue(reader.getPassages().isEmpty())
    }

    /**
     * Verifies that caches built for a different embedding model are rejected even when dimensions match.
     */
    @Test
    fun testLoadIndexRejectsEmbeddingModelMismatch() {
        val cacheDir = tempDir.resolve("model-cache")
        Files.createDirectories(cacheDir)
        val vectorPayload =
            buildString {
                append("[[")
                repeat(1536) { index ->
                    if (index > 0) append(",")
                    append(if (index == 0) "1.0" else "0.0")
                }
                append("]]")
            }
        Files.writeString(cacheDir.resolve("vectors.json"), vectorPayload)
        Files.writeString(
            cacheDir.resolve("metadata.json"),
            """
            {
              "passages": ["model-specific document"],
              "metadata": [{}],
              "embedding_model": "text-embedding-3-small",
              "vector_dimension": 1536
            }
            """.trimIndent(),
        )

        val reader = VectorStoreRetriever(embeddingModel = "text-embedding-ada-002", embeddingApiKey = "reader-key")
        val loaded = reader.loadIndex(cacheDir.toString())

        assertFalse(loaded)
        assertTrue(reader.getPassages().isEmpty())
    }

    /**
     * Verifies that custom embedding overrides can load caches without assuming the hash default dimension.
     */
    @Test
    fun testLoadIndexAllowsCustomOverrideWithoutExplicitDimension() {
        val cacheDir = tempDir.resolve("override-cache")
        val customEmbedding =
            object : EmbeddingModel {
                override fun encode(text: String): DoubleArray = doubleArrayOf(1.0, 2.0, 3.0)
            }
        val writer = VectorStoreRetriever(embeddingModelOverride = customEmbedding)
        writer.indexCorpus(listOf("override document"))
        assertTrue(writer.saveIndex(cacheDir.toString()))

        val reader = VectorStoreRetriever(embeddingModelOverride = customEmbedding)
        val loaded = reader.loadIndex(cacheDir.toString())

        assertTrue(loaded)
        assertEquals(listOf("override document"), reader.getPassages())
    }
}
