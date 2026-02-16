package causalrag.generator.promptbuilder

import gg.jte.TemplateEngine
import gg.jte.output.StringOutput
import gg.jte.resolve.DirectoryCodeResolver
import gg.jte.ContentType
import java.nio.file.Files
import java.nio.file.Path

class JtePromptRenderer(private val templatesDir: String?) {
    data class Model(
        val query: String,
        val passages: List<String>,
        val causalPaths: List<List<String>>?,
        val causalGraphSummary: String?,
        val pathSummaries: List<String>?,
    )

    fun render(templateStyle: String, model: Model): String? {
        val dir = resolveTemplatesDir() ?: return null
        val templateName = "causal_prompt_${templateStyle}.jte"
        val fallbackName = "causal_prompt.jte"
        val primary = dir.resolve(templateName)
        val fallback = dir.resolve(fallbackName)
        val name =
            when {
                Files.exists(primary) -> templateName
                Files.exists(fallback) -> fallbackName
                else -> return null
            }

        val engine = TemplateEngine.create(DirectoryCodeResolver(dir), ContentType.Plain)
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
        return if (Files.exists(local)) local else null
    }
}
