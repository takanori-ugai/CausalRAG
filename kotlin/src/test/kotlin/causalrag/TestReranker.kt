package causalrag

import causalrag.causalgraph.retriever.CausalPathRetriever
import causalrag.reranker.BaseReranker
import causalrag.reranker.CausalPathReranker
import causalrag.retriever.Bm25Retriever
import causalrag.retriever.HybridRetriever
import causalrag.retriever.VectorStoreRetriever
import io.mockk.every
import io.mockk.mockk
import io.mockk.verify
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Tests causal reranking behavior and score ordering.
 */
class TestReranker {
    private lateinit var mockRetriever: CausalPathRetriever
    private lateinit var reranker: CausalPathReranker

    private val testPaths =
        listOf(
            listOf("climate change", "rising sea levels", "coastal flooding"),
            listOf("deforestation", "carbon capture reduction", "CO2 increase"),
        )

    private val testNodes =
        listOf(
            "climate change",
            "rising sea levels",
            "coastal flooding",
            "deforestation",
            "carbon capture",
            "CO2",
        )

    private val testDocs =
        listOf(
            "Climate patterns have been changing in recent decades.",
            "Rising sea levels caused by climate change lead to coastal flooding.",
            "Deforestation reduces the ability of forests to capture carbon.",
            "Coral reefs are affected by ocean temperature changes.",
            "Coastal cities are implementing flood protection measures.",
        )

    /**
     * Initializes the mocked retriever and reranker under test.
     */
    @BeforeTest
    fun setUp() {
        mockRetriever = mockk()
        every { mockRetriever.retrievePathNodes(any(), any(), any(), any()) } returns testNodes
        every { mockRetriever.retrievePaths(any(), any(), any(), any()) } returns testPaths
        reranker = CausalPathReranker(mockRetriever)
    }

    /**
     * Verifies baseline reranking behavior for a simple overlap scorer.
     */
    @Test
    fun testBaseReranker() {
        class SimpleReranker : BaseReranker("simple") {
            override fun rerank(
                query: String,
                candidates: List<String>,
                metadata: List<Map<String, Any>>?,
            ): List<Pair<String, Double>> {
                val queryWords =
                    query
                        .lowercase()
                        .split(Regex("\\s+"))
                        .filter { it.isNotBlank() }
                        .toSet()
                return candidates
                    .map { doc ->
                        val docWords =
                            doc
                                .lowercase()
                                .split(Regex("\\s+"))
                                .filter { it.isNotBlank() }
                                .toSet()
                        doc to queryWords.intersect(docWords).size.toDouble()
                    }.sortedByDescending { it.second }
            }
        }

        val reranker = SimpleReranker()
        val ranked = reranker.rerank("climate change effects", testDocs)
        assertEquals(testDocs.size, ranked.size)
        assertTrue(
            ranked
                .first()
                .first
                .lowercase()
                .contains("climate"),
        )
    }

    /**
     * Verifies that causal path signals promote relevant passages.
     */
    @Test
    fun testCausalPathReranker() {
        val query = "How does climate change affect coastal areas?"
        val rankedDocs = reranker.rerank(query, testDocs)

        verify { mockRetriever.retrievePaths(query, any(), any(), any()) }
        assertTrue(
            rankedDocs
                .first()
                .first
                .lowercase()
                .contains("rising sea levels"),
        )
        assertTrue(
            rankedDocs
                .first()
                .first
                .lowercase()
                .contains("coastal flooding"),
        )
    }

    /**
     * Verifies that documents containing the full causal chain rank highest.
     */
    @Test
    fun testDocumentScoring() {
        every { mockRetriever.retrievePaths(any(), any(), any(), any()) } returns
            listOf(listOf("climate change", "rising sea levels", "coastal flooding"))

        val docs =
            listOf(
                "Climate change is causing rising sea levels.",
                "Coastal flooding is a problem in many cities.",
                "Climate change leads to rising sea levels, which causes coastal flooding.",
            )
        val ranked = reranker.rerank("What causes coastal flooding?", docs)
        assertTrue(ranked.first().first.contains("climate change leads to rising sea levels", ignoreCase = true))
    }

    /**
     * Verifies that custom weights still favor passages preserving the full chain.
     */
    @Test
    fun testRerankingWithWeights() {
        val weighted =
            CausalPathReranker(
                mockRetriever,
                nodeMatchWeight = 1.0,
                pathMatchWeight = 3.0,
            )

        every { mockRetriever.retrievePaths(any(), any(), any(), any()) } returns
            listOf(listOf("climate change", "rising sea levels", "coastal flooding"))

        val docs =
            listOf(
                "Climate change and flooding are environmental issues.",
                "Rising sea levels are causing coastal flooding worldwide.",
                "Climate change leads to rising sea levels, causing coastal flooding.",
            )

        val ranked = weighted.rerank("How does climate change cause flooding?", docs)
        assertTrue(ranked.first().first.contains("climate change leads to rising sea levels", ignoreCase = true))
    }

    /**
     * Verifies that later repeated mentions can still satisfy causal ordering.
     */
    @Test
    fun testPathStructureUsesLaterRepeatedMentions() {
        every { mockRetriever.retrievePathNodes(any(), any(), any(), any()) } returns
            listOf("cause", "effect")
        every { mockRetriever.retrievePaths(any(), any(), any(), any()) } returns
            listOf(listOf("cause", "effect"))

        val docs =
            listOf(
                "Effect appears first. Later, cause appears and then effect appears again.",
                "Effect appears first and cause appears later.",
            )

        val ranked = reranker.rerank("test repeated mentions", docs)

        assertEquals(docs[0], ranked.first().first)
        assertTrue(ranked[0].second > ranked[1].second)
    }

    /**
     * Verifies that explanations only report causal relationships when the passage preserves order.
     */
    @Test
    fun testExplanationRequiresPreservedOrder() {
        every { mockRetriever.retrievePathNodes(any(), any(), any(), any()) } returns
            listOf("cause", "effect")
        every { mockRetriever.retrievePaths(any(), any(), any(), any()) } returns
            listOf(listOf("cause", "effect"))

        val explanation = reranker.getExplanation("test explanation", "Effect appears before cause.", null)

        assertTrue(explanation.contains("Matched concepts (2/2):"))
        assertTrue(explanation.contains("No causal relationships preserved."))
        assertTrue(explanation.contains("cause -> effect").not())
    }

    /**
     * Verifies that BM25 returns no passages when a query has no lexical matches.
     */
    @Test
    fun testBm25ReturnsNoResultsForMiss() {
        val retriever = Bm25Retriever()
        retriever.indexCorpus(testDocs)

        val results = retriever.retrieve("quasar nebula pulsar", topK = 3)

        assertTrue(results.isEmpty())
    }

    /**
     * Verifies that BM25-only candidates can still be returned when semantic retrieval is empty.
     */
    @Test
    fun testHybridRetrieverSupportsBm25Fallback() {
        val vectorRetriever = mockk<VectorStoreRetriever>()
        val graphRetriever = mockk<CausalPathRetriever>()
        val bm25Retriever = Bm25Retriever()
        bm25Retriever.indexCorpus(testDocs)

        every { vectorRetriever.searchWithScores(any(), any(), any()) } returns emptyList()
        every { graphRetriever.retrievePathNodes(any(), any(), any(), any()) } returns emptyList()
        every { graphRetriever.retrievePaths(any(), any(), any(), any()) } returns emptyList()

        val retriever =
            HybridRetriever(
                vectorRetriever = vectorRetriever,
                graphRetriever = graphRetriever,
                semanticWeight = 0.0,
                causalWeight = 0.0,
                bm25Weight = 1.0,
                bm25Retriever = bm25Retriever,
            )

        val results = retriever.retrieveWithDetails("climate change", topK = 3)

        assertTrue(results.isNotEmpty())
        assertTrue(results.any { (it["passage"] as String).contains("Climate", ignoreCase = true) })
    }
}
