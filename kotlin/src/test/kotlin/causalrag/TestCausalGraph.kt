package causalrag

import causalrag.causalgraph.builder.CausalGraphBuilder
import causalrag.causalgraph.retriever.CausalPathRetriever
import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Tests causal graph construction, persistence, and path retrieval.
 */
class TestCausalGraph {
    private lateinit var builder: CausalGraphBuilder
    private lateinit var retriever: CausalPathRetriever
    private lateinit var tempFile: Path

    private val testDocuments =
        listOf(
            "Climate change causes rising sea levels.",
            "Rising sea levels leads to coastal flooding.",
            "Coastal flooding causes population displacement.",
        )

    /**
     * Initializes a small graph and retriever for each test.
     */
    @BeforeTest
    fun setUp() {
        builder = CausalGraphBuilder(extractorMethod = "rule")
        builder.indexDocuments(testDocuments)
        retriever = CausalPathRetriever(builder)
        tempFile = Files.createTempFile("causalrag-graph", ".json")
    }

    /**
     * Cleans up temporary files created during testing.
     */
    @AfterTest
    fun tearDown() {
        Files.deleteIfExists(tempFile)
    }

    /**
     * Verifies that causal edges are extracted into the graph.
     */
    @Test
    fun testGraphConstruction() {
        val graph = builder.getGraph()
        assertTrue(graph.numberOfNodes() > 0)
        assertTrue(graph.numberOfEdges() > 0)

        assertTrue(hasRelation("climate change", "rising sea levels"))
        assertTrue(hasRelation("rising sea levels", "coastal flooding"))
        assertTrue(hasRelation("coastal flooding", "population displacement"))
    }

    /**
     * Verifies that graphs can be saved and loaded without losing structure.
     */
    @Test
    fun testSaveLoadGraph() {
        builder.save(tempFile.toString())
        assertTrue(Files.exists(tempFile))

        val newBuilder = CausalGraphBuilder(extractorMethod = "rule")
        val loaded = newBuilder.load(tempFile.toString())
        assertTrue(loaded)

        val original = builder.getGraph()
        val reloaded = newBuilder.getGraph()
        assertTrue(reloaded.numberOfNodes() > 0)
        assertTrue(reloaded.numberOfEdges() > 0)
        assertTrue(hasRelation("climate change", "rising sea levels", newBuilder))
        assertEquals(original.numberOfNodes(), reloaded.numberOfNodes(), "Node count mismatch after reload")
        assertEquals(original.numberOfEdges(), reloaded.numberOfEdges(), "Edge count mismatch after reload")
    }

    /**
     * Verifies that the retriever can recover causal paths from the graph.
     */
    @Test
    fun testCausalPathRetrieval() {
        val paths = retriever.retrievePaths("What are the effects of climate change?", maxPaths = 3)
        assertTrue(paths.isNotEmpty())
        val hasClimatePath =
            paths.any { path ->
                path.isNotEmpty() && path.first().lowercase().contains("climate change")
            }
        assertTrue(hasClimatePath, "Should find paths starting with climate change")
    }

    /**
     * Verifies that relevant nodes are returned for a query.
     */
    @Test
    fun testQueryRelevance() {
        val nodes =
            retriever.retrieveNodes(
                "How does climate change affect coastal infrastructure?",
                topK = 10,
                threshold = 0.0,
            )
        assertTrue(nodes.isNotEmpty())
        val labels = nodes.map { (nodeId, _) -> builder.nodeText[nodeId] ?: nodeId }
        val required = listOf("climate change", "coastal flooding")
        for (term in required) {
            assertTrue(labels.any { it.lowercase().contains(term) }, "Should find $term in relevant nodes")
        }
    }

    private fun hasRelation(
        cause: String,
        effect: String,
        targetBuilder: CausalGraphBuilder = builder,
    ): Boolean {
        val graph = targetBuilder.getGraph()
        val nodeText = targetBuilder.nodeText
        return graph.edges().any { edge ->
            val fromText = nodeText[edge.from] ?: edge.from
            val toText = nodeText[edge.to] ?: edge.to
            fromText.equals(cause, ignoreCase = true) && toText.equals(effect, ignoreCase = true)
        }
    }
}
