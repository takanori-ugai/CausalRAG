package causalrag.generator.promptbuilder

import causalrag.generator.llm.LLMInterface
import io.github.oshai.kotlinlogging.KotlinLogging
import java.nio.file.Files
import java.nio.file.Path

private val logger = KotlinLogging.logger {}

@Suppress("TooGenericExceptionCaught")
class PromptBuilder(
    private val templateStyle: String = "detailed",
    private val llmInterface: LLMInterface? = null,
    private val templatesDir: String? = null,
) {
    private val jteRenderer = JtePromptRenderer(templatesDir)

    fun buildPrompt(
        query: String,
        passages: List<String>,
        causalPaths: List<List<String>>? = null,
        causalNodes: List<String>? = null,
        causalGraphSummary: String? = null,
    ): String {
        val normalizedPaths = causalPaths ?: causalNodes?.let { listOf(it) }
        val pathSummaries =
            if (!normalizedPaths.isNullOrEmpty() && llmInterface != null) {
                generateCausalSummaries(normalizedPaths, query)
            } else {
                null
            }
        val jteRendered =
            jteRenderer.render(
                templateStyle,
                JtePromptRenderer.Model(
                    query = query,
                    passages = passages,
                    causalPaths = normalizedPaths,
                    causalGraphSummary = causalGraphSummary,
                    pathSummaries = pathSummaries,
                ),
            )
        if (jteRendered != null) {
            return jteRendered
        }
        val template = loadTemplate(templateStyle)
        if (template != null) {
            val renderer = TemplateRenderer()
            return renderer.render(
                template,
                TemplateRenderer.Context(
                    query = query,
                    passages = passages,
                    causalPaths = normalizedPaths,
                    causalGraphSummary = causalGraphSummary,
                    pathSummaries = pathSummaries,
                ),
            )
        }
        return when (templateStyle) {
            "basic" -> buildBasicPrompt(query, passages, normalizedPaths, causalNodes, pathSummaries)
            "structured" -> buildStructuredPrompt(query, passages, normalizedPaths, causalNodes, causalGraphSummary, pathSummaries)
            "chain_of_thought" -> buildCotPrompt(query, passages, normalizedPaths, causalNodes, causalGraphSummary, pathSummaries)
            else -> buildDetailedPrompt(query, passages, normalizedPaths, causalNodes, causalGraphSummary, pathSummaries)
        }
    }

    private fun generateCausalSummaries(
        causalPaths: List<List<String>>,
        query: String,
    ): List<String> {
        if (llmInterface == null) return emptyList()
        val maxPathsToSummarize = 5
        return try {
            val selectedPaths = causalPaths.take(maxPathsToSummarize)
            val allPathsText = selectedPaths.joinToString("\n") { it.joinToString(" -> ") }
            val overviewPrompt = """Summarize the following causal relationships as they relate to: "$query"

Causal paths:
$allPathsText

Provide a concise summary (1-2 sentences) that captures the key causal mechanisms:"""
            val overview = llmInterface.generate(overviewPrompt, temperature = 0.3, maxTokens = 150)
            val summaries = mutableListOf<String>()
            summaries.add(overview.trim())
            for (path in selectedPaths) {
                val pathText = path.joinToString(" -> ")
                if (path.size <= 3) {
                    summaries.add(rewriteAsNaturalLanguage(pathText))
                } else {
                    val summaryPrompt = """Convert this causal relationship path into a natural language explanation:

Causal path: $pathText

Write a concise explanation using simple natural language that maintains all the causal relationships:"""
                    summaries.add(llmInterface.generate(summaryPrompt, temperature = 0.3, maxTokens = 100).trim())
                }
            }
            summaries
        } catch (ex: Exception) {
            logger.error(ex) { "Error generating causal summaries" }
            emptyList()
        }
    }

    private fun rewriteAsNaturalLanguage(pathText: String): String {
        val text = pathText.replace(" -> ", " leads to ")
        val capitalized = text.replaceFirstChar { if (it.isLowerCase()) it.titlecase() else it.toString() }
        return if (capitalized.endsWith('.')) capitalized else "$capitalized."
    }

    private fun buildBasicPrompt(
        query: String,
        passages: List<String>,
        causalPaths: List<List<String>>?,
        causalNodes: List<String>?,
        pathSummaries: List<String>?,
    ): String {
        val builder = StringBuilder()
        builder.append("Answer the following question using the provided context.\n\n")
        if (!causalNodes.isNullOrEmpty()) {
            builder.append("Causal concepts: ").append(causalNodes.joinToString(", ")).append("\n\n")
        }
        if (!causalPaths.isNullOrEmpty()) {
            builder.append("Causal relationships:\n")
            if (!pathSummaries.isNullOrEmpty()) {
                if (pathSummaries.size > 1) {
                    builder.append("Overview: ").append(pathSummaries[0]).append("\n\n")
                }
                causalPaths.forEachIndexed { idx, path ->
                    builder
                        .append("[")
                        .append(idx + 1)
                        .append("] ")
                        .append(path.joinToString(" -> "))
                        .append("\n")
                    if (idx + 1 < pathSummaries.size) {
                        builder.append("   Explanation: ").append(pathSummaries[idx + 1]).append("\n")
                    }
                }
            } else {
                causalPaths.forEachIndexed { idx, path ->
                    builder
                        .append("[")
                        .append(idx + 1)
                        .append("] ")
                        .append(path.joinToString(" -> "))
                        .append("\n")
                }
            }
            builder.append("\n")
        }
        builder.append("Context:\n")
        passages.forEachIndexed { idx, p ->
            builder
                .append("[")
                .append(idx + 1)
                .append("] ")
                .append(p)
                .append("\n")
        }
        builder.append("\nQuestion: ").append(query).append("\nAnswer:")
        return builder.toString()
    }

    private fun buildDetailedPrompt(
        query: String,
        passages: List<String>,
        causalPaths: List<List<String>>?,
        causalNodes: List<String>?,
        causalGraphSummary: String?,
        pathSummaries: List<String>?,
    ): String {
        val builder = StringBuilder()
        builder.append(
            """
            |You are a causal reasoning assistant. Answer the following question using:
            |1. The provided context passages
            |2. The causal relationships between concepts
            |3. Your understanding of how causes lead to effects
            |
            |Ensure your answer reflects the causal mechanisms described in the context.
            """.trimMargin(),
        )
        if (!causalNodes.isNullOrEmpty()) {
            builder.append("\nCausal concepts: ").append(causalNodes.joinToString(", ")).append("\n")
        }
        if (!causalPaths.isNullOrEmpty()) {
            if (!pathSummaries.isNullOrEmpty()) {
                builder.append("\nCausal overview: ").append(pathSummaries[0]).append("\n")
            }
            builder.append("\nRelevant causal relationships:\n")
            causalPaths.forEachIndexed { idx, path ->
                builder
                    .append("[")
                    .append(idx + 1)
                    .append("] ")
                    .append(path.joinToString(" -> "))
                    .append("\n")
                if (!pathSummaries.isNullOrEmpty() && idx + 1 < pathSummaries.size) {
                    builder.append("   In other words: ").append(pathSummaries[idx + 1]).append("\n")
                }
            }
            if (!causalGraphSummary.isNullOrBlank()) {
                builder.append("\nCausal graph summary: ").append(causalGraphSummary).append("\n")
            }
        }
        builder.append("\nContext passages:\n")
        passages.forEachIndexed { idx, p ->
            builder
                .append("[")
                .append(idx + 1)
                .append("] ")
                .append(p.trim())
                .append("\n")
        }
        builder
            .append("\nQuestion: ")
            .append(query)
            .append("\n\nAnswer (explain the causal relationships that lead to your conclusion):")
        return builder.toString()
    }

    private fun buildStructuredPrompt(
        query: String,
        passages: List<String>,
        causalPaths: List<List<String>>?,
        causalNodes: List<String>?,
        causalGraphSummary: String?,
        pathSummaries: List<String>?,
    ): String {
        val builder = StringBuilder()
        builder.append(
            """
            |You are a causal reasoning assistant that explains complex relationships between concepts.
            |For the following question, provide a structured answer that:
            |1. Identifies the key causal factors involved
            |2. Explains how these factors relate through causal mechanisms
            |3. Provides a final answer that follows from this causal chain
            |
            |Use the provided context passages and causal relationship information.
            """.trimMargin(),
        )
        if (!causalNodes.isNullOrEmpty()) {
            builder.append("\nCausal concepts: ").append(causalNodes.joinToString(", ")).append("\n")
        }
        if (!causalPaths.isNullOrEmpty()) {
            if (!pathSummaries.isNullOrEmpty()) {
                builder.append("\nSUMMARY OF CAUSAL MECHANISMS: ").append(pathSummaries[0]).append("\n")
            }
            builder.append("\nRelevant causal pathways:\n")
            causalPaths.forEachIndexed { idx, path ->
                builder
                    .append("[")
                    .append(idx + 1)
                    .append("] ")
                    .append(path.joinToString(" -> "))
                    .append("\n")
                if (!pathSummaries.isNullOrEmpty() && idx + 1 < pathSummaries.size) {
                    builder.append("   Natural language: ").append(pathSummaries[idx + 1]).append("\n")
                }
            }
            if (!causalGraphSummary.isNullOrBlank()) {
                builder.append("\nCausal graph structure: ").append(causalGraphSummary).append("\n")
            }
            builder.append("\nImportant: Use these causal pathways to structure your reasoning.")
        }
        builder.append("\nContext passages:\n")
        passages.forEachIndexed { idx, p ->
            builder
                .append("[")
                .append(idx + 1)
                .append("] ")
                .append(p.trim())
                .append("\n")
        }
        builder.append(
            """
            |Question: $query
            |
            |Your structured causal answer:
            |1. Causal factors:
            |2. Causal mechanisms:
            |3. Conclusion:
            """.trimMargin(),
        )
        return builder.toString()
    }

    private fun buildCotPrompt(
        query: String,
        passages: List<String>,
        causalPaths: List<List<String>>?,
        causalNodes: List<String>?,
        causalGraphSummary: String?,
        pathSummaries: List<String>?,
    ): String {
        val builder = StringBuilder()
        builder.append(
            """
            |You are an expert in causal reasoning who answers complex questions by tracing causal mechanisms.
            |For the question below, think step-by-step through the causal chains involved:
            |
            |1. First identify all key concepts from the question
            |2. For each causal relationship relevant to these concepts:
            |   - Examine what causes what
            |   - Consider the strength and direction of the relationship
            |   - Look for mediators and moderators of the relationship
            |3. Then trace through the most plausible causal paths
            |4. Finally, synthesize these relationships into a cohesive explanation
            |
            |Reference only information from the provided context and causal relationships.
            """.trimMargin(),
        )
        if (!causalNodes.isNullOrEmpty()) {
            builder.append("\nCausal concepts: ").append(causalNodes.joinToString(", ")).append("\n")
        }
        if (!causalPaths.isNullOrEmpty()) {
            if (!pathSummaries.isNullOrEmpty()) {
                builder.append("\nKEY INSIGHT ABOUT THESE CAUSAL MECHANISMS: ").append(pathSummaries[0]).append("\n")
            }
            builder.append("\nCAUSAL RELATIONSHIPS TO CONSIDER:\n")
            causalPaths.forEachIndexed { idx, path ->
                builder
                    .append("[")
                    .append(idx + 1)
                    .append("] ")
                    .append(path.joinToString(" -> "))
                    .append("\n")
                if (!pathSummaries.isNullOrEmpty() && idx + 1 < pathSummaries.size) {
                    builder.append("   Explanation: ").append(pathSummaries[idx + 1]).append("\n")
                }
            }
            if (!causalGraphSummary.isNullOrBlank()) {
                builder.append("\nGlobal causal structure: ").append(causalGraphSummary).append("\n")
            }
        }
        builder.append("\nREFERENCE CONTEXTS:\n")
        passages.forEachIndexed { idx, p ->
            builder
                .append("[")
                .append(idx + 1)
                .append("] ")
                .append(p.trim())
                .append("\n")
        }
        builder.append(
            """
            |QUESTION: $query
            |
            |STEP-BY-STEP REASONING:
            |1) Key concepts in this question are:
            |2) Relevant causal relationships from the context:
            |3) Tracing the causal chain:
            |4) Therefore, the answer is:
            """.trimMargin(),
        )
        return builder.toString()
    }

    private fun loadTemplate(style: String): String? {
        require(style.matches(Regex("[a-zA-Z0-9_]+"))) { "Invalid template style: $style" }
        val filename = "causal_prompt_$style.txt"
        val fallback = "causal_prompt.txt"
        val fromDir =
            templatesDir?.let { dir ->
                val primary = Path.of(dir, filename)
                val fallbackPath = Path.of(dir, fallback)
                when {
                    Files.exists(primary) -> Files.readString(primary)
                    Files.exists(fallbackPath) -> Files.readString(fallbackPath)
                    else -> null
                }
            }
        if (fromDir != null) return fromDir

        val classLoader = javaClass.classLoader
        return classLoader
            .getResourceAsStream("causalrag/templates/$filename")
            ?.bufferedReader()
            ?.use { it.readText() }
            ?: classLoader
                .getResourceAsStream("causalrag/templates/$fallback")
                ?.bufferedReader()
                ?.use { it.readText() }
    }
}

fun buildPrompt(
    query: String,
    passages: List<String>,
    causalPaths: List<List<String>>? = null,
    causalNodes: List<String>? = null,
    causalGraphSummary: String? = null,
    templateStyle: String = "detailed",
    llmInterface: LLMInterface? = null,
    templatesDir: String? = null,
): String {
    val builder =
        PromptBuilder(
            templateStyle = templateStyle,
            llmInterface = llmInterface,
            templatesDir = templatesDir,
        )
    return builder.buildPrompt(query, passages, causalPaths, causalNodes, causalGraphSummary)
}
