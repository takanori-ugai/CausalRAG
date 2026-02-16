package causalrag

import causalrag.causalgraph.retriever.CausalPathRetriever
import causalrag.reranker.BaseReranker
import causalrag.reranker.CausalPathReranker
import io.mockk.every
import io.mockk.mockk
import io.mockk.verify
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

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

    @BeforeTest
    fun setUp() {
        mockRetriever = mockk()
        every { mockRetriever.retrievePathNodes(any(), any(), any(), any()) } returns testNodes
        every { mockRetriever.retrievePaths(any(), any(), any(), any()) } returns testPaths
        reranker = CausalPathReranker(mockRetriever)
    }

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

    @Test
    fun testCausalPathReranker() {
        val query = "How does climate change affect coastal areas?"
        val rankedDocs = reranker.rerank(query, testDocs)

        verify { mockRetriever.retrievePaths(query, maxPaths = 3, minPathLength = 2, maxPathLength = 4) }
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
}
