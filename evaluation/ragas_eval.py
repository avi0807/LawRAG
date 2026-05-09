from typing import List, Dict
import json
from openai import OpenAI
from rich.console import Console
from rich.table import Table
from config import cfg

console = Console()


class SimpleRAGASEvaluator:

    def __init__(self):
        self.model = OpenAI(base_url=cfg.ollama_base_url, api_key=cfg.ollama_api_key)

    def _llm(self, prompt: str) -> str:
        response = self.model.chat.completions.create(
            model=cfg.ollama_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0,
        )
        return response.choices[0].message.content.strip()

    def evaluate_faithfulness(self, answer: str, context_chunks: List[str]) -> float:
        context = "\n\n".join(context_chunks[:5])

        result = self._llm(f"""Given this context and answer, evaluate faithfulness.

Context:
{context}

Answer:
{answer}

List each factual claim in the answer and whether the context supports it.
Return JSON:
{{
  "claims": [
    {{"claim": "...", "supported": true/false}}
  ]
}}
Only JSON, no explanation.""")

        try:
            clean = result.strip().strip("```json").strip("```")
            data = json.loads(clean)
            claims = data.get("claims", [])
            if not claims:
                return 1.0
            supported = sum(1 for c in claims if c.get("supported", False))
            return round(supported / len(claims), 3)
        except (json.JSONDecodeError, ZeroDivisionError):
            return 0.5

    def evaluate_answer_relevancy(self, question: str, answer: str) -> float:
        result = self._llm(f"""Rate how well this answer addresses the question.

Question: {question}

Answer: {answer}

Return JSON: {{"score": 0.0, "reason": "brief reason"}}
Score 0.0-1.0 where:
1.0 = perfectly addresses the question
0.5 = partially relevant
0.0 = completely off-topic
Only JSON.""")

        try:
            clean = result.strip().strip("```json").strip("```")
            data = json.loads(clean)
            return round(float(data.get("score", 0.5)), 3)
        except (json.JSONDecodeError, ValueError):
            return 0.5

    def evaluate_context_precision(self, question: str, context_chunks: List[str]) -> float:
        if not context_chunks:
            return 0.0

        relevant_count = 0
        for chunk in context_chunks[:cfg.top_k]:
            verdict = self._llm(f"""Is this context useful for answering the question?

Question: {question}
Context: {chunk[:500]}

Return JSON: {{"relevant": true/false}}
Only JSON.""")
            try:
                clean = verdict.strip().strip("```json").strip("```")
                data = json.loads(clean)
                if data.get("relevant", False):
                    relevant_count += 1
            except json.JSONDecodeError:
                pass

        return round(relevant_count / len(context_chunks[:cfg.top_k]), 3)

    def evaluate_sample(self, question: str, answer: str, context_chunks: List[str]) -> Dict[str, float]:
        console.print(f"\n[bold]Evaluating:[/bold] {question[:60]}...")

        faithfulness = self.evaluate_faithfulness(answer, context_chunks)
        console.print(f"  Faithfulness:       {faithfulness:.3f}")

        relevancy = self.evaluate_answer_relevancy(question, answer)
        console.print(f"  Answer relevancy:   {relevancy:.3f}")

        precision = self.evaluate_context_precision(question, context_chunks)
        console.print(f"  Context precision:  {precision:.3f}")

        overall = round((faithfulness + relevancy + precision) / 3, 3)
        console.print(f"  [bold]Overall score:      {overall:.3f}[/bold]")

        return {
            "faithfulness": faithfulness,
            "answer_relevancy": relevancy,
            "context_precision": precision,
            "overall": overall,
        }

    def evaluate_batch(self, test_cases: List[Dict]) -> Dict[str, float]:
        all_scores = []
        for case in test_cases:
            scores = self.evaluate_sample(
                question=case["question"],
                answer=case["answer"],
                context_chunks=case["context_chunks"],
            )
            all_scores.append(scores)

        metrics = ["faithfulness", "answer_relevancy", "context_precision", "overall"]
        avg_scores = {
            m: round(sum(s[m] for s in all_scores) / len(all_scores), 3)
            for m in metrics
        }

        table = Table(title="RAGAS Evaluation Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Score", style="bold")
        table.add_column("Interpretation", style="dim")

        interpretations = {
            "faithfulness":      "> 0.8 = low hallucination",
            "answer_relevancy":  "> 0.8 = on-topic answers",
            "context_precision": "> 0.7 = good retrieval",
            "overall":           "> 0.75 = production ready",
        }

        for metric, score in avg_scores.items():
            color = "green" if score > 0.75 else "yellow" if score > 0.5 else "red"
            table.add_row(
                metric,
                f"[{color}]{score:.3f}[/{color}]",
                interpretations[metric],
            )

        console.print(table)
        return avg_scores