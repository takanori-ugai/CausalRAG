package causalrag.examples

import io.github.oshai.kotlinlogging.KotlinLogging

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
    logger.info { "Loading evaluation data from $evalDataPath" }
    logger.info { "Running evaluation..." }
    val result =
        CliUtils.runEvaluation(
            config =
                CliUtils.EvalRunConfig(
                    evalDataPath = evalDataPath,
                    outputDir = outputDir,
                    modelName = modelName,
                    evalModel = evalModel,
                    embeddingModel = embeddingModel,
                    apiKey = apiKey,
                    provider = provider,
                    indexDir = indexDir,
                ),
            warn = { msg -> logger.warn { msg } },
            error = { msg -> logger.error { msg } },
        )
            ?: return

    logger.info { "Evaluation complete! Summary:" }
    result.results.metrics.forEach { (metric, score) ->
        logger.info { "  $metric: ${"%.4f".format(score)}" }
    }

    logger.info { "Detailed results saved to ${result.outputDir}" }
}
