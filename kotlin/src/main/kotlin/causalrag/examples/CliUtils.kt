package causalrag.examples

import causalrag.CausalRAGPipeline
import causalrag.evaluation.CausalEvaluator
import causalrag.evaluation.EvalExample
import causalrag.evaluation.EvaluationResult
import causalrag.generator.llm.LLMInterface
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.serialization.SerializationException
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import java.nio.file.Files
import java.nio.file.Path

object CliUtils {
    private val logger = KotlinLogging.logger {}

    data class EvalRunConfig(
        val evalDataPath: String,
        val outputDir: String,
        val modelName: String,
        val evalModel: String?,
        val embeddingModel: String,
        val apiKey: String?,
        val provider: String,
        val indexDir: String?,
    )

    data class EvalRunResult(
        val results: EvaluationResult,
        val outputDir: String,
    )

    fun parseOptions(args: List<String>): Map<String, String> {
        val options = mutableMapOf<String, String>()
        var i = 0
        while (i < args.size) {
            val arg = args[i]
            if (arg.startsWith("-")) {
                val key = arg.trimStart('-')
                val value = args.getOrNull(i + 1)
                if (value != null && (!value.startsWith("-") || isNegativeNumber(value))) {
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

    private fun isNegativeNumber(value: String): Boolean =
        value.startsWith("-") &&
            value.matches(Regex("^-?(\\d+\\.?\\d*|\\.\\d+)([eE][+-]?\\d+)?$"))

    fun loadEvaluationData(filepath: String): List<EvalExample> {
        val json = Json { ignoreUnknownKeys = true }
        val text =
            try {
                Files.readString(Path.of(filepath))
            } catch (ex: java.io.IOException) {
                logger.warn(ex) { "Failed to read evaluation data from $filepath" }
                return emptyList()
            }
        val root =
            try {
                json.parseToJsonElement(text)
            } catch (ex: SerializationException) {
                logger.warn(ex) { "Failed to parse evaluation data from $filepath" }
                return emptyList()
            }
        if (root !is JsonArray) return emptyList()
        return root.mapNotNull { element ->
            val obj = element as? JsonObject ?: return@mapNotNull null
            val question = (obj["question"] as? JsonPrimitive)?.content ?: return@mapNotNull null
            val groundTruth = (obj["ground_truth"] as? JsonPrimitive)?.content
            EvalExample(question = question, groundTruth = groundTruth)
        }
    }

    fun runEvaluation(
        config: EvalRunConfig,
        warn: (String) -> Unit,
        error: (String) -> Unit,
    ): EvalRunResult? {
        val pipeline = CausalRAGPipeline(modelName = config.modelName, embeddingModel = config.embeddingModel)
        if (config.indexDir != null) {
            val loaded = pipeline.load(config.indexDir)
            if (!loaded) {
                warn("Failed to load index from ${config.indexDir}; evaluation may lack causal paths.")
            }
        } else {
            warn("No --index provided; evaluation will run without a prebuilt graph.")
        }

        val evalData = loadEvaluationData(config.evalDataPath)
        if (evalData.isEmpty()) {
            error("No evaluation examples found in ${config.evalDataPath}")
            return null
        }

        val llm =
            LLMInterface(
                modelName = config.evalModel ?: config.modelName,
                apiKey = config.apiKey,
                provider = config.provider,
                systemMessage = "You are an expert evaluator assessing the quality of answers to questions.",
            )

        Files.createDirectories(Path.of(config.outputDir))

        val metrics =
            listOf(
                "causal_consistency",
                "causal_completeness",
                "answer_quality",
            )

        val results =
            CausalEvaluator.evaluatePipeline(
                pipeline = pipeline,
                evalData = evalData,
                metrics = metrics,
                llmInterface = llm,
                resultsDir = config.outputDir,
            )

        return EvalRunResult(results, config.outputDir)
    }
}
