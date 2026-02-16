package causalrag.evaluation

import causalrag.CausalRAGPipeline
import causalrag.generator.llm.LLMInterface
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import java.nio.file.Files
import java.nio.file.Path
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import kotlin.math.max
import kotlin.math.min

private val logger = KotlinLogging.logger {}

@Serializable
data class EvaluationResult(
    val metrics: Map<String, Double>,
    val detailedScores: Map<String, List<Double>>,
    val errorAnalysis: Map<String, String>? = null,
    val rawEvaluations: Map<String, JsonElement>? = null,
)

data class EvalExample(
    val question: String,
    val groundTruth: String? = null,
)

class CausalEvaluator(
    private val llmInterface: LLMInterface? = null,
    metrics: List<String>? = null,
    private val resultsDir: String? = null,
) {
    private val defaultMetrics =
        listOf(
            "causal_consistency",
            "causal_completeness",
            "answer_quality",
        )

    private val metricsToUse = metrics ?: defaultMetrics

    fun evaluate(
        questions: List<String>,
        answers: List<String>,
        contexts: List<List<String>>,
        causalPaths: List<List<List<String>>>? = null,
        groundTruths: List<String>? = null,
    ): EvaluationResult {
        require(questions.size == answers.size && questions.size == contexts.size) {
            "Number of questions, answers, and contexts must match"
        }
        if (causalPaths != null) {
            require(questions.size == causalPaths.size) {
                "Number of questions and causal paths must match"
            }
        }

        val allMetrics = mutableMapOf<String, Double>()
        val detailedScores = mutableMapOf<String, List<Double>>()
        val errorAnalysis = mutableMapOf<String, String>()
        val rawEvaluations = mutableMapOf<String, JsonElement>()

        if (llmInterface != null) {
            val causalResults =
                evaluateCausalReasoning(
                    questions,
                    answers,
                    contexts,
                    causalPaths ?: List(questions.size) { emptyList() },
                )
            allMetrics.putAll(causalResults.metrics)
            detailedScores.putAll(causalResults.detailedScores)
            causalResults.errorAnalysis?.let { errorAnalysis.putAll(it) }
            causalResults.rawEvaluations?.let { rawEvaluations.putAll(it) }

            val llmResults = evaluateAnswerQuality(questions, answers, contexts, groundTruths)
            allMetrics.putAll(llmResults.metrics)
            detailedScores.putAll(llmResults.detailedScores)
            llmResults.errorAnalysis?.let { errorAnalysis.putAll(it) }
            llmResults.rawEvaluations?.let { rawEvaluations.putAll(it) }
        }

        val result =
            EvaluationResult(
                metrics = allMetrics,
                detailedScores = detailedScores,
                errorAnalysis = if (errorAnalysis.isEmpty()) null else errorAnalysis,
                rawEvaluations = if (rawEvaluations.isEmpty()) null else rawEvaluations,
            )

        if (resultsDir != null) {
            saveResults(result, resultsDir)
        }

        return result
    }

    private fun evaluateCausalReasoning(
        questions: List<String>,
        answers: List<String>,
        contexts: List<List<String>>,
        causalPaths: List<List<List<String>>>,
    ): EvaluationResult {
        val metrics = mutableMapOf<String, Double>()
        val detailed = mutableMapOf<String, List<Double>>()
        val errors = mutableMapOf<String, String>()

        if (llmInterface == null) {
            return EvaluationResult(metrics, detailed, errors)
        }

        try {
            if ("causal_consistency" in metricsToUse) {
                val scores =
                    questions.indices.map { idx ->
                        val paths = causalPaths[idx]
                        if (paths.isEmpty()) return@map 1.0
                        val pathsText = paths.joinToString("\n") { it.joinToString(" -> ") }
                        val prompt =
                            """Evaluate if the answer respects the causal relationships provided.

Question: ${questions[idx]}

Causal relationships that should be respected:
$pathsText

Answer to evaluate:
${answers[idx]}

On a scale of 0-10, how well does the answer respect these causal relationships?
Provide your rating as a number from 0-10, followed by a brief explanation.
Rating:""".trimIndent()
                        val response = llmInterface.generate(prompt, temperature = 0.1)
                        parseScore(response)
                    }
                metrics["causal_consistency"] = scores.average()
                detailed["causal_consistency"] = scores
            }

            if ("causal_completeness" in metricsToUse) {
                val scores =
                    questions.indices.map { idx ->
                        val paths = causalPaths[idx]
                        if (paths.isEmpty()) return@map 1.0
                        val factors = paths.flatten().toSet()
                        val factorsText = factors.joinToString("\n") { "- $it" }
                        val prompt =
                            """Evaluate if the answer addresses all important causal factors.

Question: ${questions[idx]}

Important causal factors that should be addressed:
$factorsText

Answer to evaluate:
${answers[idx]}

On a scale of 0-10, how completely does the answer address these important causal factors?
Provide your rating as a number from 0-10, followed by a brief explanation.
Rating:""".trimIndent()
                        val response = llmInterface.generate(prompt, temperature = 0.1)
                        parseScore(response)
                    }
                metrics["causal_completeness"] = scores.average()
                detailed["causal_completeness"] = scores
            }
        } catch (ex: RuntimeException) {
            logger.error(ex) { "Error in causal evaluation" }
            errors["causal_evaluation"] = ex.message ?: "Unknown error"
        }

        return EvaluationResult(metrics, detailed, if (errors.isEmpty()) null else errors)
    }

    private fun evaluateAnswerQuality(
        questions: List<String>,
        answers: List<String>,
        contexts: List<List<String>>,
        groundTruths: List<String>?,
    ): EvaluationResult {
        val metrics = mutableMapOf<String, Double>()
        val detailed = mutableMapOf<String, List<Double>>()
        val errors = mutableMapOf<String, String>()

        if (llmInterface == null || "answer_quality" !in metricsToUse) {
            return EvaluationResult(metrics, detailed, if (errors.isEmpty()) null else errors)
        }

        try {
            val scores =
                questions.indices.map { idx ->
                    val contextText =
                        contexts[idx].mapIndexed { j, c -> "[${j + 1}] $c" }.joinToString("\n")
                    val gtText =
                        groundTruths?.getOrNull(idx)?.let { "\nGround truth answer:\n$it" } ?: ""
                    val prompt =
                        """Evaluate the quality of this answer based on the question and provided context.

Question: ${questions[idx]}

Context:
$contextText$gtText

Answer to evaluate:
${answers[idx]}

Rate the answer on a scale of 0-10 based on accuracy, completeness, conciseness, and coherence.
Provide your overall rating as a number from 0-10.
Overall rating:""".trimIndent()
                    val response = llmInterface.generate(prompt, temperature = 0.1)
                    parseScore(response, defaultScore = 0.7)
                }
            metrics["answer_quality"] = scores.average()
            detailed["answer_quality"] = scores
        } catch (ex: RuntimeException) {
            logger.error(ex) { "Error in LLM evaluation" }
            errors["llm_evaluation"] = ex.message ?: "Unknown error"
        }

        return EvaluationResult(metrics, detailed, if (errors.isEmpty()) null else errors)
    }

    private fun parseScore(response: String, defaultScore: Double = 0.5): Double {
        val firstLine = response.trim().lineSequence().firstOrNull().orEmpty()
        val match = Regex("(\\d+(\\.\\d+)?)").find(firstLine)
        val value = match?.groupValues?.getOrNull(1)?.toDoubleOrNull() ?: defaultScore
        val normalized = value / 10.0
        return min(max(normalized, 0.0), 1.0)
    }

    private fun saveResults(result: EvaluationResult, dir: String) {
        try {
            val outputDir = Path.of(dir)
            Files.createDirectories(outputDir)
            val timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"))
            val json = Json { prettyPrint = true }

            val metricsFile = outputDir.resolve("metrics_summary_$timestamp.json")
            Files.writeString(metricsFile, json.encodeToString(MapSerializer, result.metrics))

            val detailedFile = outputDir.resolve("detailed_scores_$timestamp.json")
            Files.writeString(detailedFile, json.encodeToString(MapListSerializer, result.detailedScores))

            result.errorAnalysis?.let {
                val errorsFile = outputDir.resolve("error_analysis_$timestamp.json")
                Files.writeString(errorsFile, json.encodeToString(MapStringSerializer, it))
            }

            val reportFile = outputDir.resolve("evaluation_report_$timestamp.md")
            Files.writeString(reportFile, buildReport(result))

            logger.info { "Evaluation results saved to $outputDir" }
        } catch (ex: java.io.IOException) {
            logger.error(ex) { "Error saving evaluation results" }
        } catch (ex: RuntimeException) {
            logger.error(ex) { "Error saving evaluation results" }
        }
    }

    private fun buildReport(result: EvaluationResult): String {
        val builder = StringBuilder()
        builder.append("# CausalRAG Evaluation Report\n\n")
        builder.append("## Metrics Summary\n\n")
        result.metrics.forEach { (metric, score) ->
            builder.append("- **$metric:** ${"%.4f".format(score)}\n")
        }
        builder.append("\n## Metric Details\n\n")
        result.detailedScores.forEach { (metric, scores) ->
            val avg = scores.average()
            val sorted = scores.sorted()
            val median = if (sorted.isEmpty()) 0.0 else sorted[sorted.size / 2]
            val minVal = scores.minOrNull() ?: 0.0
            val maxVal = scores.maxOrNull() ?: 0.0
            builder.append("### $metric\n")
            builder.append("- **Average:** ${"%.4f".format(avg)}\n")
            builder.append("- **Median:** ${"%.4f".format(median)}\n")
            builder.append("- **Min:** ${"%.4f".format(minVal)}\n")
            builder.append("- **Max:** ${"%.4f".format(maxVal)}\n\n")
        }
        return builder.toString()
    }

    companion object {
        fun evaluatePipeline(
            pipeline: CausalRAGPipeline,
            evalData: List<EvalExample>,
            metrics: List<String>? = null,
            llmInterface: LLMInterface? = null,
            resultsDir: String? = null,
        ): EvaluationResult {
            val evaluator =
                CausalEvaluator(
                    llmInterface = llmInterface ?: pipeline.llm,
                    metrics = metrics,
                    resultsDir = resultsDir,
                )

            val questions = evalData.map { it.question }
            val groundTruths = evalData.mapNotNull { it.groundTruth }
            val hasGroundTruths = groundTruths.isNotEmpty()

            val answers = mutableListOf<String>()
            val contexts = mutableListOf<List<String>>()
            val causalPaths = mutableListOf<List<List<String>>>()

            for (question in questions) {
                val answer = pipeline.run(question)
                answers.add(answer)
                val context = pipeline.hybridRetriever.retrieve(question, topK = 5).map { it as String }
                contexts.add(context)
                val paths = pipeline.graphRetriever.retrievePaths(question, maxPaths = 3)
                causalPaths.add(paths)
            }

            return evaluator.evaluate(
                questions = questions,
                answers = answers,
                contexts = contexts,
                causalPaths = causalPaths,
                groundTruths = if (hasGroundTruths) groundTruths else null,
            )
        }
    }
}

private val MapSerializer =
    object : kotlinx.serialization.KSerializer<Map<String, Double>> {
        override val descriptor = kotlinx.serialization.descriptors.buildClassSerialDescriptor("MapStringDouble")
        override fun serialize(encoder: kotlinx.serialization.encoding.Encoder, value: Map<String, Double>) {
            val jsonEncoder = encoder as? kotlinx.serialization.json.JsonEncoder
            val jsonObject = buildJsonObject { value.forEach { put(it.key, JsonPrimitive(it.value)) } }
            jsonEncoder?.encodeJsonElement(jsonObject)
        }

        override fun deserialize(decoder: kotlinx.serialization.encoding.Decoder): Map<String, Double> {
            return emptyMap()
        }
    }

private val MapListSerializer =
    object : kotlinx.serialization.KSerializer<Map<String, List<Double>>> {
        override val descriptor = kotlinx.serialization.descriptors.buildClassSerialDescriptor("MapStringListDouble")
        override fun serialize(encoder: kotlinx.serialization.encoding.Encoder, value: Map<String, List<Double>>) {
            val jsonEncoder = encoder as? kotlinx.serialization.json.JsonEncoder
            val jsonObject =
                buildJsonObject {
                    value.forEach { (k, v) ->
                        put(k, JsonArray(v.map { JsonPrimitive(it) }))
                    }
                }
            jsonEncoder?.encodeJsonElement(jsonObject)
        }

        override fun deserialize(decoder: kotlinx.serialization.encoding.Decoder): Map<String, List<Double>> {
            return emptyMap()
        }
    }

private val MapStringSerializer =
    object : kotlinx.serialization.KSerializer<Map<String, String>> {
        override val descriptor = kotlinx.serialization.descriptors.buildClassSerialDescriptor("MapStringString")
        override fun serialize(encoder: kotlinx.serialization.encoding.Encoder, value: Map<String, String>) {
            val jsonEncoder = encoder as? kotlinx.serialization.json.JsonEncoder
            val jsonObject = buildJsonObject { value.forEach { put(it.key, JsonPrimitive(it.value)) } }
            jsonEncoder?.encodeJsonElement(jsonObject)
        }

        override fun deserialize(decoder: kotlinx.serialization.encoding.Decoder): Map<String, String> {
            return emptyMap()
        }
    }
