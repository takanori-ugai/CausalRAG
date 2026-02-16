package causalrag

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class TestPipeline {
    private lateinit var tempDir: Path
    private lateinit var configPath: Path

    private val testDocs =
        listOf(
            "Climate change causes rising sea levels.",
            "Rising sea levels leads to coastal flooding.",
            "Coastal flooding causes population displacement.",
        )

    @BeforeTest
    fun setUp() {
        tempDir = Files.createTempDirectory("causalrag-test")
        configPath = tempDir.resolve("causalrag-test-config.json")
        writeTestConfig(configPath)
    }

    @AfterTest
    fun tearDown() {
        tempDir.toFile().deleteRecursively()
    }

    @Test
    fun testPipelineInit() {
        val pipeline = CausalRAGPipeline(configPath = configPath.toString())
        assertNotNull(pipeline)
        assertNotNull(pipeline.graphBuilder)
        assertNotNull(pipeline.vectorRetriever)
    }

    @Test
    fun testDocumentIndexing() {
        val pipeline = CausalRAGPipeline(configPath = configPath.toString())
        pipeline.index(testDocs)
        val graph = pipeline.graphBuilder.getGraph()
        assertTrue(graph.numberOfNodes() > 0)
    }

    @Test
    fun testQueryExecution() {
        val pipeline = CausalRAGPipeline(configPath = configPath.toString())
        pipeline.index(testDocs)
        val answer = pipeline.run("What causes coastal flooding?")
        assertTrue(answer.isNotBlank())
    }

    @Test
    fun testSaveAndLoad() {
        val pipeline1 = CausalRAGPipeline(configPath = configPath.toString())
        pipeline1.index(testDocs)

        val saveDir = tempDir.resolve("causalrag_index")
        Files.createDirectories(saveDir)
        val saved = pipeline1.save(saveDir.toString())
        assertTrue(saved)

        val pipeline2 = CausalRAGPipeline(configPath = configPath.toString())
        val loaded = pipeline2.load(saveDir.toString())
        assertTrue(loaded)

        val answer = pipeline2.run("What is climate change?")
        assertTrue(answer.isNotBlank())
    }

    private fun writeTestConfig(path: Path) {
        val json =
            JsonObject(
                mapOf(
                    "modelName" to JsonPrimitive("gpt-4o-mini"),
                    "embeddingModel" to JsonPrimitive("text-embedding-3-small"),
                    "llmProvider" to JsonPrimitive("mock"),
                    "llmApiKey" to JsonPrimitive(""),
                    "llmBaseUrl" to JsonNull,
                    "embeddingApiKey" to JsonPrimitive(""),
                    "graphPath" to JsonNull,
                    "indexPath" to JsonNull,
                    "templateStyle" to JsonPrimitive("detailed"),
                ),
            )
        val content =
            Json { prettyPrint = true }.encodeToString(
                JsonElement.serializer(),
                json,
            )
        Files.writeString(path, content)
    }
}
