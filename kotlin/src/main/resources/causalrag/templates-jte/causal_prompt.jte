@import java.util.List
@param String query
@param List<String> passages
@param List<List<String>> causalPaths
@param String causalGraphSummary
@param List<String> pathSummaries

You are a causal reasoning assistant. Answer the following question using:
1. The provided context passages
2. The causal relationships between concepts
3. Your understanding of how causes lead to effects

Ensure your answer reflects the causal mechanisms described in the context.

@if(causalPaths != null && !causalPaths.isEmpty())
Relevant causal relationships:
@for(int i = 0; i < causalPaths.size(); i++)
[${i + 1}] ${String.join(" → ", causalPaths.get(i))}
@endfor
@endif

Context passages:
@for(int i = 0; i < passages.size(); i++)
[${i + 1}] ${passages.get(i)}
@endfor

Question: ${query}

Answer (explain the causal relationships that lead to your conclusion):
