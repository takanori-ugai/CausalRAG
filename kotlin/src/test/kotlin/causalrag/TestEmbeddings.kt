package causalrag

import causalrag.utils.LangChain4jEmbeddingModel
import causalrag.utils.SimpleHashEmbedding
import causalrag.utils.resolveDefaultModelName
import dev.langchain4j.data.embedding.Embedding
import dev.langchain4j.data.segment.TextSegment
import dev.langchain4j.model.output.Response
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertIs
import kotlin.test.assertTrue

/**
 * Tests embedding utility validation behavior.
 */
class TestEmbeddings {
    @Test
    fun testCreateDefaultResolutionRemapsLocalDefaultWhenApiKeyPresent() {
        val resolved = resolveDefaultModelName("all-MiniLM-L6-v2", hasApiKey = true)

        assertEquals("text-embedding-3-small", resolved)
    }

    @Test
    fun testCreateDefaultFallsBackToSimpleHashEmbeddingWithoutApiKey() {
        val embeddingModel = causalrag.utils.EmbeddingModelFactory.createDefault("all-MiniLM-L6-v2", apiKey = null)

        assertIs<SimpleHashEmbedding>(embeddingModel)
    }

    /**
     * Verifies that hash embeddings reject non-positive dimensions.
     */
    @Test
    fun testSimpleHashEmbeddingRejectsNonPositiveDimension() {
        val zeroError =
            assertFailsWith<IllegalArgumentException> {
                SimpleHashEmbedding(0)
            }
        val negativeError =
            assertFailsWith<IllegalArgumentException> {
                SimpleHashEmbedding(-1)
            }

        assertTrue(zeroError.message!!.contains("dimension must be > 0"))
        assertTrue(negativeError.message!!.contains("dimension must be > 0"))
    }

    @Test
    fun testSimpleHashEmbeddingProducesNonZeroVectorForJapaneseText() {
        val embedding = SimpleHashEmbedding(dimension = 32).encode("日本語の埋め込みをテストします")

        assertTrue(embedding.any { it != 0.0 })
    }

    @Test
    fun testLangChain4jEmbeddingModelUsesDelegateBatchEncoding() {
        val delegate =
            object : dev.langchain4j.model.embedding.EmbeddingModel {
                var embedAllCallCount = 0
                var lastBatchTexts: List<String> = emptyList()

                override fun embedAll(textSegments: List<TextSegment>): Response<List<Embedding>> {
                    embedAllCallCount += 1
                    lastBatchTexts = textSegments.map(TextSegment::text)
                    val embeddings =
                        textSegments.mapIndexed { index, segment ->
                            Embedding(floatArrayOf(segment.text().length.toFloat(), index.toFloat()))
                        }
                    return Response.from(embeddings)
                }
            }

        val adapter = LangChain4jEmbeddingModel(delegate)
        val encoded = adapter.encodeAll(listOf("alpha", "beta"))

        assertEquals(1, delegate.embedAllCallCount)
        assertEquals(listOf("alpha", "beta"), delegate.lastBatchTexts)
        assertEquals(listOf(5.0, 0.0), encoded[0].toList())
        assertEquals(listOf(4.0, 1.0), encoded[1].toList())
    }
}
