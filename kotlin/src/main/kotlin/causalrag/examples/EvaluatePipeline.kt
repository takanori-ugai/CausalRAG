package causalrag.examples

import causalrag.CausalRAGPipeline
import causalrag.evaluation.CausalEvaluator
import causalrag.evaluation.EvalExample
import causalrag.generator.llm.LLMInterface
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import java.nio.file.Files
import java.nio.file.Path

private val logger = KotlinLogging.logger {}

fun main(args: Array<String>) {
    val options = parseOptions(args.toList())
    val evalDataPath = options["eval-data"] ?: run {
        println("Missing required --eval-data")
        return
    }
    val outputDir = options["output-dir"] ?: "./eval_results"
    val modelName = options["model-name"] ?: "gpt-4"
    val evalModel = options["eval-model"]
    val embeddingModel = options["embedding-model"] ?: "text-embedding-3-small"
    val apiKey = options["api-key"]
    val provider = options["provider"] ?: "openai"
    val indexDir = options["index"]

    logger.info { "Initializing pipeline..." }
    val pipeline = CausalRAGPipeline(modelName = modelName, embeddingModel = embeddingModel)
    if (indexDir != null) {
        val loaded = pipeline.load(indexDir)
        if (!loaded) {
            logger.warn { "Failed to load index from $indexDir; evaluation may lack causal paths." }
        }
    } else {
        logger.warn { "No --index provided; evaluation will run without a prebuilt graph." }
    }

    logger.info { "Loading evaluation data from $evalDataPath" }
    val evalData = loadEvaluationData(evalDataPath)

    val llm =
        LLMInterface(
            modelName = evalModel ?: modelName,
            apiKey = apiKey,
            provider = provider,
            systemMessage = "You are an expert evaluator assessing the quality of answers to questions.",
        )

    Files.createDirectories(Path.of(outputDir))

    val metrics =
        listOf(
            "causal_consistency",
            "causal_completeness",
            "answer_quality",
        )

    logger.info { "Running evaluation..." }
    val results =
        CausalEvaluator.evaluatePipeline(
            pipeline = pipeline,
            evalData = evalData,
            metrics = metrics,
            llmInterface = llm,
            resultsDir = outputDir,
        )

    logger.info { "Evaluation complete! Summary:" }
    results.metrics.forEach { (metric, score) ->
        logger.info { "  $metric: ${"%.4f".format(score)}" }
    }

    logger.info { "Detailed results saved to $outputDir" }
}

private fun loadEvaluationData(filepath: String): List<EvalExample> {
    val json = Json { ignoreUnknownKeys = true }
    val text = Files.readString(Path.of(filepath))
    val root = json.parseToJsonElement(text)
    if (root !is JsonArray) return emptyList()
    return root.mapNotNull { element ->
        val obj = element as? JsonObject ?: return@mapNotNull null
        val question = (obj["question"] as? JsonPrimitive)?.content ?: return@mapNotNull null
        val groundTruth = (obj["ground_truth"] as? JsonPrimitive)?.content
        EvalExample(question = question, groundTruth = groundTruth)
    }
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
        } else {
            i += 1
        }
    }
    return options
}
