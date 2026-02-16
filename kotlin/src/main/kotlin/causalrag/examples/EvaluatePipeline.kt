package causalrag.examples

import causalrag.CausalRAGPipeline
import causalrag.evaluation.CausalEvaluator
import causalrag.evaluation.EvalExample
import causalrag.generator.llm.LLMInterface
import io.github.oshai.kotlinlogging.KotlinLogging
import java.nio.file.Files
import java.nio.file.Path

private val logger = KotlinLogging.logger {}

fun main(args: Array<String>) {
    val options = CliUtils.parseOptions(args.toList())
    val evalDataPath =
        options["eval-data"] ?: run {
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
    val evalData = CliUtils.loadEvaluationData(evalDataPath)
    if (evalData.isEmpty()) {
        logger.error { "No evaluation examples found in $evalDataPath" }
        return
    }

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
