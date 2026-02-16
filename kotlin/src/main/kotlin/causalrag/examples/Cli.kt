package causalrag.examples

import causalrag.CausalRAGPipeline
import causalrag.evaluation.CausalEvaluator
import causalrag.evaluation.EvalExample
import causalrag.generator.llm.LLMInterface
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Path
import kotlin.io.path.isDirectory
import kotlin.io.path.name
import kotlin.streams.toList

private const val VERSION = "0.0.1"

fun main(args: Array<String>) {
    if (args.isEmpty()) {
        printUsage()
        return
    }

    if (args.contains("--version")) {
        println("CausalRAG Kotlin version $VERSION")
        return
    }

    when (args.first()) {
        "index" -> handleIndex(args.drop(1))
        "query" -> handleQuery(args.drop(1))
        "serve" -> handleServe()
        "evaluate" -> handleEvaluate(args.drop(1))
        else -> printUsage()
    }
}

private fun handleIndex(args: List<String>) {
    val opts = parseOptions(args)
    val input = opts["input"] ?: opts["i"]
    val output = opts["output"] ?: opts["o"]
    val model = opts["model"]
    val config = opts["config"]

    if (input == null || output == null) {
        println("Missing required --input/-i or --output/-o")
        return
    }

    val documents = loadDocuments(input)
    if (documents.isEmpty()) {
        println("No documents found to index")
        return
    }

    val pipeline =
        when {
            config != null -> CausalRAGPipeline(configPath = config)
            model != null -> CausalRAGPipeline(embeddingModel = model)
            else -> CausalRAGPipeline()
        }

    println("Indexing ${documents.size} documents...")
    pipeline.index(documents)
    val saved = pipeline.save(output)
    if (saved) {
        println("Saved index to $output")
    } else {
        println("Indexing complete but failed to save to $output")
    }
}

private fun handleQuery(args: List<String>) {
    val opts = parseOptions(args)
    val indexDir = opts["index"] ?: opts["i"]
    val query = opts["query"] ?: opts["q"]
    val model = opts["model"]
    val topK = (opts["top-k"] ?: "5").toIntOrNull() ?: 5
    val config = opts["config"]

    if (indexDir == null || query == null) {
        println("Missing required --index/-i or --query/-q")
        return
    }

    val pipeline =
        when {
            config != null -> CausalRAGPipeline(configPath = config)
            model != null -> CausalRAGPipeline(modelName = model)
            else -> CausalRAGPipeline()
        }

    if (!pipeline.load(indexDir)) {
        println("Failed to load index from $indexDir")
        return
    }

    println("\n" + "=".repeat(80))
    println("Query: $query")
    println("=".repeat(80))

    val answer = pipeline.run(query, topK = topK)
    println("\nAnswer: $answer")

    println("\nSupporting Context:")
    val context = pipeline.hybridRetriever.retrieve(query, topK = topK).map { it as String }
    context.forEachIndexed { idx, ctx ->
        val preview = if (ctx.length > 200) ctx.substring(0, 200) + "..." else ctx
        println("[${idx + 1}] $preview")
    }

    val causalPaths = pipeline.graphRetriever.retrievePaths(query, maxPaths = 3)
    if (causalPaths.isNotEmpty()) {
        println("\nRelevant Causal Pathways:")
        causalPaths.forEachIndexed { idx, path ->
            println("[${idx + 1}] ${path.joinToString(" -> ")}")
        }
    }
}

private fun handleServe() {
    println("Serve is not implemented in the Kotlin port yet.")
}

private fun handleEvaluate(args: List<String>) {
    val opts = parseOptions(args)
    val evalData = opts["eval-data"]
    val outputDir = opts["output-dir"] ?: "./eval_results"
    val modelName = opts["model-name"] ?: "gpt-4"
    val evalModel = opts["eval-model"]
    val embeddingModel = opts["embedding-model"] ?: "text-embedding-3-small"
    val apiKey = opts["api-key"]
    val provider = opts["provider"] ?: "openai"
    val indexDir = opts["index"]

    if (evalData == null) {
        println("Missing required --eval-data")
        return
    }

    val pipeline = CausalRAGPipeline(modelName = modelName, embeddingModel = embeddingModel)
    if (indexDir != null) {
        val loaded = pipeline.load(indexDir)
        if (!loaded) {
            println("Failed to load index from $indexDir; evaluation may lack causal paths.")
        }
    } else {
        println("No --index provided; evaluation will run without a prebuilt graph.")
    }
    val llm =
        LLMInterface(
            modelName = evalModel ?: modelName,
            apiKey = apiKey,
            provider = provider,
            systemMessage = "You are an expert evaluator assessing the quality of answers to questions.",
        )

    val evalExamples = loadEvaluationData(evalData)
    if (evalExamples.isEmpty()) {
        println("No evaluation examples found in $evalData")
        return
    }

    val metrics =
        listOf(
            "causal_consistency",
            "causal_completeness",
            "answer_quality",
        )

    val results =
        CausalEvaluator.evaluatePipeline(
            pipeline = pipeline,
            evalData = evalExamples,
            metrics = metrics,
            llmInterface = llm,
            resultsDir = outputDir,
        )

    println("Evaluation complete! Summary:")
    results.metrics.forEach { (metric, score) ->
        println("  $metric: ${"%.4f".format(score)}")
    }
    println("Detailed results saved to $outputDir")
}

private fun parseOptions(args: List<String>): Map<String, String> {
    val options = mutableMapOf<String, String>()
    var i = 0
    while (i < args.size) {
        val arg = args[i]
        if (arg.startsWith("--")) {
            val key = arg.removePrefix("--")
            val value = args.getOrNull(i + 1)
            if (value != null && !value.startsWith("-")) {
                options[key] = value
                i += 2
            } else {
                options[key] = "true"
                i += 1
            }
        } else if (arg.startsWith("-")) {
            val key = arg.removePrefix("-")
            val value = args.getOrNull(i + 1)
            if (value != null && !value.startsWith("-")) {
                options[key] = value
                i += 2
            } else {
                options[key] = "true"
                i += 1
            }
        } else {
            i += 1
        }
    }
    return options
}

private fun loadDocuments(input: String): List<String> {
    val path = Path.of(input)
    return if (path.isDirectory()) {
        Files
            .walk(path)
            .filter { Files.isRegularFile(it) }
            .filter { it.name.endsWith(".txt") }
            .map { readTextFile(it) }
            .filter { it != null }
            .map { it!! }
            .toList()
    } else {
        readTextFile(path)?.let { listOf(it) } ?: emptyList()
    }
}

private fun readTextFile(path: Path): String? =
    try {
        val text = Files.readString(path, StandardCharsets.UTF_8)
        if (text.isBlank()) null else text
    } catch (_: java.io.IOException) {
        null
    }

private fun printUsage() {
    println(
        """
CausalRAG Kotlin CLI

Usage:
  cli index --input <dir|file> --output <dir> [--model <embedding_model>] [--config <path>]
  cli query --index <dir> --query <text> [--model <llm_model>] [--top-k <n>] [--config <path>]
  cli serve
  cli evaluate --eval-data <path> [--index <dir>] [--output-dir <dir>] [--model-name <llm_model>] [--eval-model <llm_model>] [--embedding-model <embedding_model>] [--api-key <key>] [--provider <name>]
  cli --version
""".trimIndent(),
    )
}

private fun loadEvaluationData(filepath: String): List<EvalExample> {
    val text = Files.readString(Path.of(filepath))
    val json = kotlinx.serialization.json.Json { ignoreUnknownKeys = true }
    val root = json.parseToJsonElement(text)
    if (root !is kotlinx.serialization.json.JsonArray) return emptyList()
    return root.mapNotNull { element ->
        val obj = element as? kotlinx.serialization.json.JsonObject ?: return@mapNotNull null
        val question = (obj["question"] as? kotlinx.serialization.json.JsonPrimitive)?.content ?: return@mapNotNull null
        val groundTruth = (obj["ground_truth"] as? kotlinx.serialization.json.JsonPrimitive)?.content
        EvalExample(question = question, groundTruth = groundTruth)
    }
}
