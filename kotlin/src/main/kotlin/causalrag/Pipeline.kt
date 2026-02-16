package causalrag

import causalrag.causalgraph.builder.CausalGraphBuilder
import causalrag.causalgraph.retriever.CausalPathRetriever
import causalrag.generator.llm.LLMInterface
import causalrag.generator.promptbuilder.buildPrompt
import causalrag.reranker.CausalPathReranker
import causalrag.retriever.Bm25Retriever
import causalrag.retriever.HybridRetriever
import causalrag.retriever.VectorStoreRetriever
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.io.IOException
import java.nio.file.Files
import java.nio.file.Path

private val logger = KotlinLogging.logger {}

@Serializable
data class PipelineConfig(
    val modelName: String? = null,
    val embeddingModel: String? = null,
    val graphPath: String? = null,
    val indexPath: String? = null,
    val llmProvider: String? = null,
    val llmApiKey: String? = null,
    val llmBaseUrl: String? = null,
    val embeddingApiKey: String? = null,
    val templateStyle: String? = null,
)

/**
 * Top-level orchestration of the CausalRAG pipeline.
 */
@Suppress("TooGenericExceptionCaught")
class CausalRAGPipeline(
    modelName: String = "gpt-4",
    embeddingModel: String = "all-MiniLM-L6-v2",
    graphPath: String? = null,
    indexPath: String? = null,
    configPath: String? = null,
) {
    private val config: PipelineConfig? = configPath?.let { loadConfig(it) }
    private val effectiveModelName = config?.modelName ?: modelName
    private val effectiveEmbeddingModel = config?.embeddingModel ?: embeddingModel
    private val effectiveGraphPath = config?.graphPath ?: graphPath
    private val effectiveIndexPath = config?.indexPath ?: indexPath
    private val effectiveLlmProvider = config?.llmProvider ?: "openai"
    private val effectiveLlmApiKey = config?.llmApiKey ?: System.getenv("OPENAI_API_KEY")
    private val effectiveLlmBaseUrl = config?.llmBaseUrl
    private val effectiveEmbeddingApiKey = config?.embeddingApiKey ?: System.getenv("OPENAI_API_KEY")
    private val effectiveTemplateStyle = config?.templateStyle ?: "detailed"

    // Core components
    val llm: LLMInterface =
        LLMInterface(
            modelName = effectiveModelName,
            provider = effectiveLlmProvider,
            apiKey = effectiveLlmApiKey,
            baseUrl = effectiveLlmBaseUrl,
        )
    val graphBuilder: CausalGraphBuilder =
        CausalGraphBuilder(
            modelName = effectiveEmbeddingModel,
            graphPath = effectiveGraphPath,
            embeddingApiKey = effectiveEmbeddingApiKey,
            extractorMethod = "hybrid",
            llmInterface = llm,
        )
    val vectorRetriever: VectorStoreRetriever =
        VectorStoreRetriever(
            embeddingModel = effectiveEmbeddingModel,
            indexPath = effectiveIndexPath,
            embeddingApiKey = effectiveEmbeddingApiKey,
        )
    val bm25Retriever: Bm25Retriever = Bm25Retriever()
    val graphRetriever: CausalPathRetriever = CausalPathRetriever(graphBuilder)
    val hybridRetriever: HybridRetriever =
        HybridRetriever(
            vectorRetriever,
            graphRetriever,
            semanticWeight = 0.4,
            causalWeight = 0.5,
            bm25Weight = 0.1,
            bm25Retriever = bm25Retriever,
        )
    val reranker: CausalPathReranker = CausalPathReranker(graphRetriever)

    /** Load configuration from file. */
    private fun loadConfig(configPath: String): PipelineConfig {
        val path = Path.of(configPath)
        require(Files.exists(path)) { "Config file not found: $configPath" }
        return try {
            val json = Json { ignoreUnknownKeys = true }
            val content = Files.readString(path)
            json.decodeFromString(PipelineConfig.serializer(), content)
        } catch (ex: IOException) {
            logger.error(ex) { "Failed to load config from $configPath" }
            throw ex
        } catch (ex: RuntimeException) {
            logger.error(ex) { "Failed to load config from $configPath" }
            throw ex
        }
    }

    /** Build graph + vector index from documents. */
    fun index(documents: List<String>) {
        graphBuilder.indexDocuments(documents)
        vectorRetriever.indexCorpus(documents)
        bm25Retriever.indexDocuments(documents)
    }

    /** Save graph and vector index to a directory. */
    fun save(dir: String): Boolean =
        try {
            val path = Path.of(dir)
            Files.createDirectories(path)
            graphBuilder.save(path.resolve("graph.json").toString())
            vectorRetriever.saveIndex(path.resolve("vector_cache").toString())
            true
        } catch (ex: IOException) {
            logger.error(ex) { "Failed to save pipeline data to $dir" }
            false
        } catch (ex: RuntimeException) {
            logger.error(ex) { "Failed to save pipeline data to $dir" }
            false
        }

    /** Load graph and vector index from a directory. */
    fun load(dir: String): Boolean =
        try {
            val path = Path.of(dir)
            val graphLoaded = graphBuilder.load(path.resolve("graph.json").toString())
            val vectorsLoaded = vectorRetriever.loadIndex(path.resolve("vector_cache").toString())
            graphLoaded && vectorsLoaded
        } catch (ex: IOException) {
            logger.error(ex) { "Failed to load pipeline data from $dir" }
            false
        } catch (ex: RuntimeException) {
            logger.error(ex) { "Failed to load pipeline data from $dir" }
            false
        }

    /** Query → Retrieval → Rerank → Prompt → Generate. */
    fun run(
        query: String,
        topK: Int = 5,
    ): String {
        // Step 1: Hybrid retrieval
        val candidates = hybridRetriever.retrieve(query, topK = topK).map { it as String }

        // Step 2: Rerank via causal path
        val reranked = reranker.rerank(query, candidates)
        val rerankedPassages = reranked.map { it.first }

        // Step 3: Build prompt with causal context
        val causalNodes = graphRetriever.retrievePathNodes(query)
        val prompt =
            buildPrompt(
                query,
                rerankedPassages.take(topK),
                causalNodes = causalNodes,
                templateStyle = effectiveTemplateStyle,
            )

        // Step 4: Generate answer
        return llm.generate(prompt)
    }
}
