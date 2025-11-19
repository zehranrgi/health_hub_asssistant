"""
RAGAS Evaluation for CVS HealthHub AI
Evaluates RAG system performance using multiple metrics
"""
import os
import sys
from typing import List, Dict
from datasets import Dataset
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.healthhub_agent import chat, vector_store
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

load_dotenv()


# Test dataset - healthcare queries with ground truth
TEST_CASES = [
    {
        "question": "What are the common side effects of Lisinopril?",
        "ground_truth": "Common side effects include dizziness, headache, persistent cough, fatigue, and low blood pressure."
    },
    {
        "question": "Can I take Aspirin with blood pressure medications?",
        "ground_truth": "Aspirin can interact with blood pressure medications. Consult your doctor before combining them."
    },
    {
        "question": "What vaccines are available at CVS?",
        "ground_truth": "CVS offers flu vaccines, COVID-19 vaccines, shingles vaccines, and other immunizations."
    },
    {
        "question": "What are the symptoms of high blood pressure?",
        "ground_truth": "High blood pressure often has no symptoms but can cause headaches, shortness of breath, and nosebleeds in severe cases."
    },
    {
        "question": "How does Metformin work for diabetes?",
        "ground_truth": "Metformin lowers blood sugar by reducing glucose production in the liver and improving insulin sensitivity."
    },
    {
        "question": "What should I do if I miss a dose of my blood pressure medication?",
        "ground_truth": "Take it as soon as you remember, but skip it if it's almost time for the next dose. Never double up."
    },
    {
        "question": "Are generic medications as effective as brand names?",
        "ground_truth": "Yes, generic medications contain the same active ingredients and are FDA-approved to be as effective as brand names."
    },
    {
        "question": "What services does CVS MinuteClinic offer?",
        "ground_truth": "MinuteClinic offers health screenings, vaccinations, minor illness treatment, and wellness services."
    },
    {
        "question": "Can I transfer my prescription to CVS?",
        "ground_truth": "Yes, you can transfer prescriptions to CVS online, through the app, or by calling the pharmacy."
    },
    {
        "question": "What is the difference between Type 1 and Type 2 diabetes?",
        "ground_truth": "Type 1 is autoimmune and requires insulin. Type 2 is related to insulin resistance and lifestyle factors."
    }
]


def run_rag_evaluation():
    """
    Run comprehensive RAGAS evaluation on the RAG system
    """
    print("🏥 CVS HealthHub AI - RAGAS Evaluation")
    print("=" * 60)

    # Prepare data
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    print("\n📊 Running evaluation on test cases...")
    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"  [{i}/{len(TEST_CASES)}] Processing: {test_case['question'][:50]}...")

        # Get agent response
        result = chat(test_case["question"])
        answer = result["response"]

        # Get contexts from vector store
        search_results = vector_store.similarity_search(
            query=test_case["question"],
            k=3
        )
        context_list = [r["content"] for r in search_results]

        # Store results
        questions.append(test_case["question"])
        answers.append(answer)
        contexts.append(context_list)
        ground_truths.append(test_case["ground_truth"])

    # Create dataset
    eval_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    print("\n✅ Data collection complete!")
    print(f"📈 Evaluating {len(TEST_CASES)} test cases with RAGAS metrics...")

    # Run evaluation
    try:
        result = evaluate(
            eval_dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            ]
        )

        # Print results
        print("\n" + "=" * 60)
        print("📊 RAGAS EVALUATION RESULTS")
        print("=" * 60)

        print(f"\n🎯 Faithfulness:        {result['faithfulness']:.4f}")
        print(f"   (How factually accurate are answers based on context)")

        print(f"\n🎯 Answer Relevancy:    {result['answer_relevancy']:.4f}")
        print(f"   (How relevant are answers to the questions)")

        print(f"\n🎯 Context Precision:   {result['context_precision']:.4f}")
        print(f"   (How precise is the retrieved context)")

        print(f"\n🎯 Context Recall:      {result['context_recall']:.4f}")
        print(f"   (How much relevant context was retrieved)")

        print("\n" + "=" * 60)

        # Calculate overall score
        overall_score = (
            result['faithfulness'] +
            result['answer_relevancy'] +
            result['context_precision'] +
            result['context_recall']
        ) / 4

        print(f"\n⭐ OVERALL SCORE: {overall_score:.4f} / 1.0")

        # Performance assessment
        if overall_score >= 0.85:
            assessment = "🟢 EXCELLENT - Production-ready RAG system"
        elif overall_score >= 0.70:
            assessment = "🟡 GOOD - Minor improvements recommended"
        elif overall_score >= 0.50:
            assessment = "🟠 FAIR - Significant improvements needed"
        else:
            assessment = "🔴 POOR - Major optimization required"

        print(f"📋 Assessment: {assessment}")
        print("=" * 60)

        # Save results
        save_results(result, overall_score)

        return result

    except Exception as e:
        print(f"\n❌ Evaluation failed: {str(e)}")
        return None


def save_results(result: Dict, overall_score: float):
    """Save evaluation results to file"""
    output_path = os.path.join(
        os.path.dirname(__file__),
        "evaluation_results.txt"
    )

    with open(output_path, "w") as f:
        f.write("CVS HealthHub AI - RAGAS Evaluation Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Faithfulness:        {result['faithfulness']:.4f}\n")
        f.write(f"Answer Relevancy:    {result['answer_relevancy']:.4f}\n")
        f.write(f"Context Precision:   {result['context_precision']:.4f}\n")
        f.write(f"Context Recall:      {result['context_recall']:.4f}\n\n")
        f.write(f"OVERALL SCORE:       {overall_score:.4f} / 1.0\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Test Cases: {len(TEST_CASES)}\n")
        f.write(f"Metrics: Faithfulness, Answer Relevancy, Context Precision, Context Recall\n")

    print(f"\n💾 Results saved to: {output_path}")


if __name__ == "__main__":
    # Check vector store
    try:
        stats = vector_store.get_collection_stats()
        print(f"\n✅ Vector Store: {stats['total_chunks']} documents loaded")
    except Exception as e:
        print(f"\n❌ Vector store error: {e}")
        print("Run: python ingestion/load_initial_data.py")
        sys.exit(1)

    # Run evaluation
    run_rag_evaluation()
