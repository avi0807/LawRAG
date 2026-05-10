from typing import List, Dict
import re
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
        if not answer.strip() or not context_chunks:
            return 0.0

        context = " ".join(context_chunks[:5]).lower()
        context_words = set(re.findall(r'\b\w{4,}\b', context))

        sentences = [s.strip() for s in re.split(r'[.!?]', answer) if len(s.strip()) > 20]
        if not sentences:
            return 0.5

        supported = 0
        for sentence in sentences:
            words = set(re.findall(r'\b\w{4,}\b', sentence.lower()))
            if not words:
                continue
            overlap = len(words & context_words) / len(words)
            if overlap > 0.25:
                supported += 1

        return round(supported / len(sentences), 3)

    def evaluate_answer_relevancy(self, question: str, answer: str) -> float:
        if not answer.strip():
            return 0.0

        q_words = set(re.findall(r'\b\w{4,}\b', question.lower()))
        a_words = set(re.findall(r'\b\w{4,}\b', answer.lower()))

        if not q_words:
            return 0.5

        overlap = len(q_words & a_words) / len(q_words)
        return round(min(overlap * 2.5, 1.0), 3)

    def evaluate_context_precision(self, question: str, context_chunks: List[str]) -> float:
        if not context_chunks:
            return 0.0

        q_words = set(re.findall(r'\b\w{4,}\b', question.lower()))
        if not q_words:
            return 0.5

        relevant_count = 0
        for chunk in context_chunks[:cfg.top_k]:
            chunk_words = set(re.findall(r'\b\w{4,}\b', chunk.lower()))
            overlap = len(q_words & chunk_words) / len(q_words)
            if overlap > 0.2:
                relevant_count += 1

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