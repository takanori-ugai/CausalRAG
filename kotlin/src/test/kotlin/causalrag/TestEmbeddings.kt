package causalrag

import causalrag.utils.SimpleHashEmbedding
import causalrag.utils.resolveDefaultModelName
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
}
