package causalrag.generator.promptbuilder

import io.github.oshai.kotlinlogging.KotlinLogging

private val logger = KotlinLogging.logger {}

class TemplateRenderer {
    data class Context(
        val query: String,
        val passages: List<String>,
        val causalPaths: List<List<String>>?,
        val causalGraphSummary: String?,
        val pathSummaries: List<String>?,
    )

    fun render(
        template: String,
        context: Context,
    ): String {
        val result = StringBuilder()
        renderSection(template, 0, context, mutableMapOf(), result)
        return result.toString()
    }

    private fun renderSection(
        template: String,
        startIndex: Int,
        context: Context,
        locals: MutableMap<String, Any>,
        output: StringBuilder,
    ): Int {
        var index = startIndex
        while (index < template.length) {
            val nextExpr = template.indexOf("{{", index)
            val nextBlock = template.indexOf("{%", index)
            val next = listOf(nextExpr, nextBlock).filter { it >= 0 }.minOrNull() ?: -1

            if (next == -1) {
                output.append(template.substring(index))
                return template.length
            }

            if (next > index) {
                output.append(template.substring(index, next))
                index = next
            }

            if (next == nextExpr) {
                val end = template.indexOf("}}", index)
                if (end == -1) {
                    output.append(template.substring(index))
                    return template.length
                }
                val expr = template.substring(index + 2, end).trim()
                output.append(evalExpression(expr, context, locals))
                index = end + 2
            } else {
                val end = template.indexOf("%}", index)
                if (end == -1) {
                    output.append(template.substring(index))
                    return template.length
                }
                val directive = template.substring(index + 2, end).trim()
                when {
                    directive.startsWith("if ") -> {
                        val condition = directive.removePrefix("if ").trim()
                        val inner = StringBuilder()
                        index = renderSection(template, end + 2, context, locals, inner)
                        if (evalCondition(condition, context, locals)) {
                            output.append(inner)
                        }
                    }

                    directive == "endif" -> {
                        return end + 2
                    }

                    directive.startsWith("for ") -> {
                        val loop = directive.removePrefix("for ").trim()
                        val parts = loop.split(" in ")
                        if (parts.size == 2) {
                            val varName = parts[0].trim()
                            val listName = parts[1].trim()
                            val list = evalList(listName, context, locals)
                            val innerTemplate = StringBuilder()
                            index = renderSection(template, end + 2, context, locals, innerTemplate)
                            for (i in list.indices) {
                                locals[varName] = list[i]
                                locals["loop.index"] = i + 1
                                output.append(renderRaw(innerTemplate.toString(), context, locals))
                            }
                            locals.remove(varName)
                            locals.remove("loop.index")
                        } else {
                            index = end + 2
                        }
                    }

                    directive == "endfor" -> {
                        return end + 2
                    }

                    else -> {
                        logger.warn { "Unknown directive: $directive" }
                        index = end + 2
                    }
                }
            }
        }
        return index
    }

    private fun renderRaw(
        template: String,
        context: Context,
        locals: MutableMap<String, Any>,
    ): String {
        val result = StringBuilder()
        renderSection(template, 0, context, locals, result)
        return result.toString()
    }

    private fun evalList(
        name: String,
        context: Context,
        locals: Map<String, Any>,
    ): List<Any> =
        when (name) {
            "passages" -> {
                context.passages
            }

            "causal_paths" -> {
                context.causalPaths ?: emptyList()
            }

            "path_summaries" -> {
                context.pathSummaries ?: emptyList()
            }

            else -> {
                val value = locals[name]
                if (value is List<*>) value.filterNotNull() else emptyList()
            }
        }

    private fun evalCondition(
        condition: String,
        context: Context,
        locals: Map<String, Any>,
    ): Boolean =
        when (condition) {
            "causal_paths" -> {
                !context.causalPaths.isNullOrEmpty()
            }

            "causal_graph_summary" -> {
                !context.causalGraphSummary.isNullOrBlank()
            }

            "path_summaries and path_summaries|length > 0" -> {
                !context.pathSummaries.isNullOrEmpty()
            }

            "path_summaries and loop.index < path_summaries|length" -> {
                val index = (locals["loop.index"] as? Int) ?: 0
                val size = context.pathSummaries?.size ?: 0
                size > 0 && index < size
            }

            else -> {
                false
            }
        }

    private fun evalExpression(
        expr: String,
        context: Context,
        locals: Map<String, Any>,
    ): String {
        if (expr.contains("|join(")) {
            val parts = expr.split("|join(")
            val listExpr = parts[0].trim()
            val separator =
                parts[1]
                    .trim()
                    .trimEnd(')')
                    .trim()
                    .trim('"', '\'')
            val list = evalList(listExpr, context, locals)
            return list.joinToString(separator) { it.toString() }
        }
        if (expr.contains("[")) {
            val base = expr.substringBefore("[").trim()
            val indexExpr = expr.substringAfter("[").substringBefore("]").trim()
            val list = evalList(base, context, locals)
            val index =
                when (indexExpr) {
                    "loop.index" -> ((locals["loop.index"] as? Int) ?: 1)
                    else -> indexExpr.toIntOrNull() ?: 0
                }
            return list.getOrNull(index)?.toString() ?: ""
        }
        return when (expr) {
            "query" -> context.query
            "causal_graph_summary" -> context.causalGraphSummary.orEmpty()
            "loop.index" -> (locals["loop.index"] as? Int)?.toString().orEmpty()
            else -> locals[expr]?.toString().orEmpty()
        }
    }
}
