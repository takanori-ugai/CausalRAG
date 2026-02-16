package causalrag.causalgraph.explainer

import causalrag.causalgraph.graph.DirectedGraph
import causalrag.causalgraph.retriever.CausalPathRetriever
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject

private val logger = KotlinLogging.logger {}

class CausalGraphExplainer(
    private val graph: DirectedGraph,
    private val nodeText: Map<String, String> = emptyMap(),
) {
    fun printPaths(
        nodes: List<String>,
        maxPathLength: Int = 5,
        includeWeights: Boolean = true,
    ): String {
        if (nodes.size < 2) return "Insufficient nodes to find paths."
        val explanation = mutableListOf<String>()
        var pathCount = 0

        for (i in nodes.indices) {
            val src = nodes[i]
            val srcName = nodeText[src] ?: src
            for (tgt in nodes.drop(i + 1)) {
                if (src == tgt) continue
                val tgtName = nodeText[tgt] ?: tgt
                for ((start, end, label) in listOf(
                    Triple(src, tgt, "Effects of $srcName on $tgtName"),
                    Triple(tgt, src, "Causes of $tgtName from $srcName"),
                )) {
                    val paths = findPaths(start, end, maxPathLength)
                    if (paths.isNotEmpty()) {
                        explanation.add("\n$label:")
                        for ((idx, path) in paths.take(3).withIndex()) {
                            val segments = mutableListOf<String>()
                            for (j in 0 until path.size - 1) {
                                val n1 = nodeText[path[j]] ?: path[j]
                                val n2 = nodeText[path[j + 1]] ?: path[j + 1]
                                val weight = graph.edgeWeight(path[j], path[j + 1])
                                if (includeWeights && weight != null) {
                                    segments.add("$n1 --(${String.format("%.2f", weight)})--> $n2")
                                } else {
                                    segments.add("$n1 --> $n2")
                                }
                            }
                            explanation.add("  Path ${idx + 1}: ${segments.joinToString(" ")}")
                            pathCount += 1
                        }
                    }
                }
            }
        }

        if (pathCount == 0) return "No paths found between the specified nodes."
        return explanation.joinToString("\n")
    }

    fun summarizeGraph(): String {
        if (graph.numberOfNodes() == 0) return "Empty causal graph (no nodes or relationships)."
        val numNodes = graph.numberOfNodes()
        val numEdges = graph.numberOfEdges()

        val summary = mutableListOf(
            "Causal Graph Summary:",
            "- Concepts: $numNodes",
            "- Relationships: $numEdges",
        )

        val outDegrees = graph.nodes().associateWith { graph.outDegree(it) }
        val inDegrees = graph.nodes().associateWith { graph.inDegree(it) }

        val topCauses = outDegrees.entries.sortedByDescending { it.value }.take(5)
        if (topCauses.isNotEmpty()) {
            summary.add("\nTop causes (most outgoing relationships):")
            for ((node, degree) in topCauses) {
                if (degree > 0) {
                    val name = nodeText[node] ?: node
                    summary.add("- $name ($degree effects)")
                }
            }
        }

        val topEffects = inDegrees.entries.sortedByDescending { it.value }.take(5)
        if (topEffects.isNotEmpty()) {
            summary.add("\nTop effects (most incoming relationships):")
            for ((node, degree) in topEffects) {
                if (degree > 0) {
                    val name = nodeText[node] ?: node
                    summary.add("- $name ($degree causes)")
                }
            }
        }

        val components = graph.weaklyConnectedComponents()
        if (components.size > 1) {
            summary.add("\nThe graph contains ${components.size} disconnected causal networks.")
            val largest = components.sortedByDescending { it.size }.take(3)
            largest.forEachIndexed { idx, component ->
                if (component.size > 1) {
                    val samples = component.take(3).map { nodeText[it] ?: it }
                    summary.add("- Network ${idx + 1}: ${component.size} concepts including ${samples.joinToString(", ")}...")
                }
            }
        }

        return summary.joinToString("\n")
    }

    fun explainQueryRelevance(query: String, retriever: CausalPathRetriever): String {
        val relevantNodes = retriever.retrieveNodes(query, topK = 5)
        if (relevantNodes.isEmpty()) {
            return "No concepts directly relevant to '$query' were found in the causal graph."
        }
        val explanation = mutableListOf("Query: '$query'\n", "Directly relevant concepts:")
        for ((nodeId, score) in relevantNodes) {
            val name = nodeText[nodeId] ?: nodeId
            explanation.add("- $name (relevance: ${String.format("%.2f", score)})")
        }
        val paths = retriever.retrievePaths(query, maxPaths = 3)
        if (paths.isNotEmpty()) {
            explanation.add("\nRelevant causal pathways:")
            paths.forEachIndexed { idx, path ->
                explanation.add("${idx + 1}. ${path.joinToString(" -> ")}")
            }
            explanation.add("\nInterpretation:")
            explanation.add("These pathways show how concepts related to '$query' influence each other through causal mechanisms.")
        }
        return explanation.joinToString("\n")
    }

    fun generateGraphVizHtml(
        highlightNodes: List<String>? = null,
        highlightEdges: List<Pair<String, String>>? = null,
    ): String {
        val json = Json { prettyPrint = false }
        val nodes =
            graph.nodes().map { node ->
                buildJsonObject {
                    put("id", JsonPrimitive(node))
                    put("name", JsonPrimitive(nodeText[node] ?: node))
                    put("highlighted", JsonPrimitive(highlightNodes?.contains(node) == true))
                }
            }
        val links =
            graph.edges().map { edge ->
                buildJsonObject {
                    put("source", JsonPrimitive(edge.from))
                    put("target", JsonPrimitive(edge.to))
                    put("weight", JsonPrimitive(edge.weight))
                    put(
                        "highlighted",
                        JsonPrimitive(highlightEdges?.contains(edge.from to edge.to) == true),
                    )
                }
            }

        val nodesJson = json.encodeToString(JsonArray.serializer(), JsonArray(nodes))
        val linksJson = json.encodeToString(JsonArray.serializer(), JsonArray(links))

        return """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Interactive Causal Graph</title>
  <script src="https://d3js.org/d3.v7.min.js"></script>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; }
    #graph-container { width: 100%; height: 800px; }
    .node { stroke: #fff; stroke-width: 1.5px; }
    .node.highlighted { stroke: #ff0000; stroke-width: 2px; }
    .link { stroke: #999; stroke-opacity: 0.6; }
    .link.highlighted { stroke: #ff0000; stroke-width: 2px; }
    .node-label { font-size: 10px; pointer-events: none; }
    .controls { padding: 10px; background: #f8f8f8; }
  </style>
</head>
<body>
  <div class="controls">
    <button id="zoom-in">+</button>
    <button id="zoom-out">-</button>
    <button id="reset">Reset</button>
    <input type="checkbox" id="show-labels" checked> <label for="show-labels">Show Labels</label>
  </div>
  <div id="graph-container"></div>
  <script>
    const graph = {"nodes": $nodesJson, "links": $linksJson};
    const width = document.getElementById('graph-container').clientWidth;
    const height = document.getElementById('graph-container').clientHeight;
    const svg = d3.select("#graph-container").append("svg")
      .attr("width", width)
      .attr("height", height);
    const g = svg.append("g");
    const zoom = d3.zoom().scaleExtent([0.1, 4]).on("zoom", (event) => g.attr("transform", event.transform));
    svg.call(zoom);
    const simulation = d3.forceSimulation(graph.nodes)
      .force("link", d3.forceLink(graph.links).id(d => d.id).distance(100))
      .force("charge", d3.forceManyBody().strength(-200))
      .force("center", d3.forceCenter(width / 2, height / 2));
    const link = g.append("g").selectAll("line")
      .data(graph.links)
      .enter().append("line")
      .attr("class", d => "link" + (d.highlighted ? " highlighted" : ""))
      .attr("stroke-width", d => Math.sqrt(d.weight) * 1.5);
    svg.append("defs").selectAll("marker")
      .data(["end"])
      .enter().append("marker")
      .attr("id", d => d)
      .attr("viewBox", "0 -5 10 10")
      .attr("refX", 25)
      .attr("refY", 0)
      .attr("markerWidth", 6)
      .attr("markerHeight", 6)
      .attr("orient", "auto")
      .append("path")
      .attr("d", "M0,-5L10,0L0,5");
    link.attr("marker-end", "url(#end)");
    const node = g.append("g").selectAll("circle")
      .data(graph.nodes)
      .enter().append("circle")
      .attr("class", d => "node" + (d.highlighted ? " highlighted" : ""))
      .attr("r", 10)
      .attr("fill", d => d.highlighted ? "#ff9999" : "#69b3a2")
      .call(d3.drag().on("start", dragstarted).on("drag", dragged).on("end", dragended));
    const label = g.append("g").selectAll("text")
      .data(graph.nodes)
      .enter().append("text")
      .attr("class", "node-label")
      .attr("text-anchor", "middle")
      .attr("dy", 3)
      .text(d => d.name.length > 20 ? d.name.substring(0, 17) + "..." : d.name);
    node.append("title").text(d => d.name);
    simulation.on("tick", () => {
      link
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);
      node
        .attr("cx", d => d.x)
        .attr("cy", d => d.y);
      label
        .attr("x", d => d.x)
        .attr("y", d => d.y - 15);
    });
    function dragstarted(event, d) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }
    function dragged(event, d) {
      d.fx = event.x;
      d.fy = event.y;
    }
    function dragended(event, d) {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }
    document.getElementById("zoom-in").addEventListener("click", () => {
      svg.transition().call(zoom.scaleBy, 1.3);
    });
    document.getElementById("zoom-out").addEventListener("click", () => {
      svg.transition().call(zoom.scaleBy, 0.7);
    });
    document.getElementById("reset").addEventListener("click", () => {
      svg.transition().call(zoom.transform, d3.zoomIdentity);
    });
    document.getElementById("show-labels").addEventListener("change", function() {
      label.style("display", this.checked ? "block" : "none");
    });
  </script>
</body>
</html>
""".trimIndent()
    }

    private fun findPaths(start: String, end: String, maxDepth: Int): List<List<String>> {
        val results = mutableListOf<List<String>>()
        fun dfs(current: String, target: String, depth: Int, path: MutableList<String>, visited: MutableSet<String>) {
            if (depth > maxDepth) return
            if (current == target) {
                results.add(path.toList())
                return
            }
            for (next in graph.successors(current)) {
                if (next in visited) continue
                visited.add(next)
                path.add(next)
                dfs(next, target, depth + 1, path, visited)
                path.removeAt(path.size - 1)
                visited.remove(next)
            }
        }
        dfs(start, end, 0, mutableListOf(start), mutableSetOf(start))
        return results
    }
}
