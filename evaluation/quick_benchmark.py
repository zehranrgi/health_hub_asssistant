"""
Quick Performance Benchmark for CVS HealthHub AI
Measures response time, accuracy, and tool usage without external dependencies
"""
import os
import sys
import time
from typing import List, Dict, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.healthhub_agent import chat, vector_store


# Benchmark test cases
BENCHMARK_QUERIES = [
    ("What are the side effects of Lisinopril?", "medication"),
    ("Can I take Aspirin with blood pressure medications?", "interaction"),
    ("What vaccines are available?", "vaccine"),
    ("What are CVS pharmacy hours?", "service"),
    ("Is my insurance accepted?", "insurance"),
    ("Tell me about Metformin", "medication"),
    ("Check interactions between Lisinopril and Aspirin", "interaction"),
    ("Do I need a flu shot?", "vaccine"),
    ("Where is the nearest CVS?", "service"),
    ("How much will my prescription cost?", "insurance")
]


def benchmark_query(question: str, category: str) -> Dict:
    """Benchmark a single query"""
    start_time = time.time()

    try:
        result = chat(question)
        end_time = time.time()

        response_time = end_time - start_time
        tool_calls = result.get("tool_calls", 0)
        response_length = len(result["response"])

        return {
            "success": True,
            "response_time": response_time,
            "tool_calls": tool_calls,
            "response_length": response_length,
            "response": result["response"]
        }

    except Exception as e:
        end_time = time.time()
        return {
            "success": False,
            "response_time": end_time - start_time,
            "error": str(e)
        }


def run_benchmark():
    """Run comprehensive benchmark"""
    print("🏥 CVS HealthHub AI - Performance Benchmark")
    print("=" * 70)

    # Check vector store
    try:
        stats = vector_store.get_collection_stats()
        print(f"\n✅ Knowledge Base: {stats['total_chunks']} documents loaded")
    except Exception as e:
        print(f"\n❌ Vector store error: {e}")
        return

    print(f"\n📊 Running {len(BENCHMARK_QUERIES)} benchmark queries...")
    print("-" * 70)

    results = []
    total_time = 0
    total_tools = 0
    successes = 0

    for i, (query, category) in enumerate(BENCHMARK_QUERIES, 1):
        print(f"\n[{i}/{len(BENCHMARK_QUERIES)}] {query[:50]}...")

        result = benchmark_query(query, category)
        results.append(result)

        if result["success"]:
            successes += 1
            total_time += result["response_time"]
            total_tools += result["tool_calls"]

            print(f"    ✅ {result['response_time']:.2f}s | Tools: {result['tool_calls']} | Chars: {result['response_length']}")
        else:
            print(f"    ❌ Failed: {result.get('error', 'Unknown error')}")

    # Calculate metrics
    print("\n" + "=" * 70)
    print("📈 BENCHMARK RESULTS")
    print("=" * 70)

    success_rate = (successes / len(BENCHMARK_QUERIES)) * 100
    avg_response_time = total_time / successes if successes > 0 else 0
    avg_tool_calls = total_tools / successes if successes > 0 else 0

    print(f"\n✅ Success Rate:           {success_rate:.1f}% ({successes}/{len(BENCHMARK_QUERIES)})")
    print(f"⚡ Avg Response Time:      {avg_response_time:.2f}s")
    print(f"🔧 Avg Tool Calls:         {avg_tool_calls:.1f}")
    print(f"📚 Knowledge Base Size:    {stats['total_chunks']} documents")

    # Performance grading
    print("\n" + "-" * 70)
    print("🎯 PERFORMANCE GRADES")
    print("-" * 70)

    # Success rate grade
    if success_rate == 100:
        sr_grade = "A+"
    elif success_rate >= 90:
        sr_grade = "A"
    elif success_rate >= 80:
        sr_grade = "B"
    else:
        sr_grade = "C"

    # Response time grade (< 3s = A, < 5s = B, < 8s = C)
    if avg_response_time < 3:
        rt_grade = "A+"
    elif avg_response_time < 5:
        rt_grade = "A"
    elif avg_response_time < 8:
        rt_grade = "B"
    else:
        rt_grade = "C"

    # Tool usage grade (1.5-2.5 = optimal)
    if 1.5 <= avg_tool_calls <= 2.5:
        tu_grade = "A+"
    elif 1 <= avg_tool_calls <= 3:
        tu_grade = "A"
    elif avg_tool_calls <= 4:
        tu_grade = "B"
    else:
        tu_grade = "C"

    print(f"  Reliability (Success Rate):  [{sr_grade}] {success_rate:.1f}%")
    print(f"  Speed (Response Time):       [{rt_grade}] {avg_response_time:.2f}s")
    print(f"  Efficiency (Tool Usage):     [{tu_grade}] {avg_tool_calls:.1f} calls")

    # Overall assessment
    grades = {"A+": 4.0, "A": 3.7, "B": 3.0, "C": 2.0}
    overall_gpa = (grades[sr_grade] + grades[rt_grade] + grades[tu_grade]) / 3

    if overall_gpa >= 3.7:
        overall = "🟢 EXCELLENT - Production-Ready"
    elif overall_gpa >= 3.0:
        overall = "🟡 GOOD - Minor Optimizations Recommended"
    else:
        overall = "🟠 FAIR - Needs Improvement"

    print(f"\n⭐ OVERALL: {overall} (GPA: {overall_gpa:.2f}/4.0)")
    print("=" * 70)

    # Save results
    save_benchmark_results(success_rate, avg_response_time, avg_tool_calls, overall_gpa)

    return results


def save_benchmark_results(success_rate, avg_time, avg_tools, gpa):
    """Save benchmark results to file"""
    output_path = os.path.join(
        os.path.dirname(__file__),
        "benchmark_results.txt"
    )

    with open(output_path, "w") as f:
        f.write("CVS HealthHub AI - Performance Benchmark Results\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Success Rate:         {success_rate:.1f}%\n")
        f.write(f"Avg Response Time:    {avg_time:.2f}s\n")
        f.write(f"Avg Tool Calls:       {avg_tools:.1f}\n")
        f.write(f"Overall GPA:          {gpa:.2f}/4.0\n\n")
        f.write(f"Test Queries:         {len(BENCHMARK_QUERIES)}\n")

    print(f"\n💾 Results saved to: {output_path}")


if __name__ == "__main__":
    run_benchmark()
