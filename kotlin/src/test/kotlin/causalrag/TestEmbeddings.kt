package causalrag

import causalrag.utils.SimpleHashEmbedding
import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Tests embedding utility validation behavior.
 */
class TestEmbeddings {
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
