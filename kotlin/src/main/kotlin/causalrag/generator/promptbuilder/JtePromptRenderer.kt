package causalrag.generator.promptbuilder

import gg.jte.ContentType
import gg.jte.TemplateEngine
import gg.jte.output.StringOutput
import gg.jte.resolve.DirectoryCodeResolver
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardCopyOption

class JtePromptRenderer(
    private val templatesDir: String?,
) {
    private var cachedEngine: TemplateEngine? = null
    private var cachedDir: Path? = null

    data class Model(
        val query: String,
        val passages: List<String>,
        val causalPaths: List<List<String>>?,
        val causalGraphSummary: String?,
        val pathSummaries: List<String>?,
    )

    private fun getEngine(dir: Path): TemplateEngine {
        val existing = cachedEngine
        if (existing != null && cachedDir == dir) {
            return existing
        }
        val engine = TemplateEngine.create(DirectoryCodeResolver(dir), ContentType.Plain)
        cachedEngine = engine
        cachedDir = dir
        return engine
    }

    fun render(
        templateStyle: String,
        model: Model,
    ): String? {
        val dir = resolveTemplatesDir() ?: return null
        val templateName = "causal_prompt_$templateStyle.jte"
        val fallbackName = "causal_prompt.jte"
        val primary = dir.resolve(templateName)
        val fallback = dir.resolve(fallbackName)
        val name =
            when {
                Files.exists(primary) -> templateName
                Files.exists(fallback) -> fallbackName
                else -> return null
            }

        val engine = getEngine(dir)
        val output = StringOutput()
        val data =
            mapOf(
                "query" to model.query,
                "passages" to model.passages,
                "causalPaths" to model.causalPaths,
                "causalGraphSummary" to model.causalGraphSummary,
                "pathSummaries" to model.pathSummaries,
            )
        engine.render(name, data, output)
        return output.toString()
    }

    private fun resolveTemplatesDir(): Path? {
        if (templatesDir != null) {
            val path = Path.of(templatesDir)
            return if (Files.exists(path)) path else null
        }
        val local = Path.of("src/main/resources/causalrag/templates-jte")
        if (Files.exists(local)) {
            return local
        }
        return extractTemplatesFromClasspath()
    }

    private fun extractTemplatesFromClasspath(): Path? {
        synchronized(cacheLock) {
            cachedTemplatesDir?.let { return it }
            val loader = JtePromptRenderer::class.java.classLoader
            val tempDir = Files.createTempDirectory("causalrag-templates-jte")
            tempDir.toFile().deleteOnExit()
            for (template in TEMPLATE_FILES) {
                val resourcePath = "causalrag/templates-jte/$template"
                val stream = loader.getResourceAsStream(resourcePath) ?: return null
                stream.use {
                    val target = tempDir.resolve(template)
                    Files.copy(it, target, StandardCopyOption.REPLACE_EXISTING)
                    target.toFile().deleteOnExit()
                }
            }
            cachedTemplatesDir = tempDir
            return tempDir
        }
    }

    private companion object {
        private val TEMPLATE_FILES =
            listOf(
                "causal_prompt.jte",
                "causal_prompt_detailed.jte",
                "causal_prompt_structured.jte",
                "causal_prompt_chain_of_thought.jte",
            )
        private val cacheLock = Any()

        @Volatile private var cachedTemplatesDir: Path? = null
    }
}
