"""
Inference script for the fine-tuned Mistral-7B query classifier
Use the LoRA adapter for real-time query classification
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from typing import Dict, Tuple
import time

# Paths
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTER_PATH = "/Users/zehranrgi/Documents/health_hub/checkpoints/lora_adapter"

# Categories
CATEGORIES = ["medication", "interactions", "vaccines", "services", "insurance"]


class QueryClassifier:
    """Fine-tuned Mistral-7B query classifier for healthcare queries"""

    def __init__(self, use_quantization: bool = True):
        """
        Initialize the classifier with LoRA adapter

        Args:
            use_quantization: Use 4-bit quantization for memory efficiency
        """
        print("📦 Loading fine-tuned Mistral-7B model...")

        # Load base model
        if use_quantization:
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                device_map="auto",
                torch_dtype=torch.float16,
            )

        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(
            self.model,
            ADAPTER_PATH,
            device_map="auto",
            torch_dtype=torch.bfloat16 if use_quantization else torch.float16,
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Set to eval mode
        self.model.eval()

        print("✅ Model loaded successfully!")
        print(f"   Model: {BASE_MODEL}")
        print(f"   Adapter: {ADAPTER_PATH}")

    def classify(self, query: str) -> Dict[str, str]:
        """
        Classify a healthcare query into one of 5 categories

        Args:
            query: User's healthcare query

        Returns:
            Dictionary with classification result and confidence
        """
        # Format the input
        prompt = f"""<|im_start|>system
You are a healthcare query classifier for CVS HealthHub AI.
Your task is to classify user queries into the correct category:
- medication: Questions about medication information, uses, dosages, side effects
- interactions: Questions about drug interactions, contraindications, compatibility
- vaccines: Questions about vaccines, immunizations, scheduling, availability
- services: Questions about pharmacy services, hours, MinuteClinic, health screenings
- insurance: Questions about insurance coverage, copay, cost, payment

Respond ONLY with the category name, nothing else.
<|im_end|>
<|im_start|>user
{query}
<|im_end|>
<|im_start|>assistant
"""

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.model.device)

        # Inference with timing
        start_time = time.time()

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=20,
                num_beams=1,
                temperature=0.7,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        inference_time = time.time() - start_time

        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract category from response
        category = self._extract_category(response)

        return {
            "query": query,
            "category": category,
            "confidence": "high" if category in CATEGORIES else "low",
            "inference_time_ms": f"{inference_time*1000:.2f}",
        }

    @staticmethod
    def _extract_category(response: str) -> str:
        """Extract category from model response"""
        response_lower = response.lower()

        for category in CATEGORIES:
            if category in response_lower:
                return category

        # Default fallback
        return "medication"


def benchmark_classifier():
    """Benchmark the fine-tuned classifier"""
    print("\n📊 Benchmarking classifier performance...")
    print("="*60)

    classifier = QueryClassifier(use_quantization=True)

    # Test queries (one per category)
    test_queries = [
        ("What are the side effects of lisinopril?", "medication"),
        ("Can I take aspirin with metformin?", "interactions"),
        ("Where can I get a flu shot?", "vaccines"),
        ("What are your pharmacy hours?", "services"),
        ("Is my insurance accepted?", "insurance"),
    ]

    total_time = 0
    correct = 0

    print("\n🔍 Classification Results:")
    print("-"*60)

    for query, expected_category in test_queries:
        result = classifier.classify(query)
        is_correct = result["category"] == expected_category

        if is_correct:
            correct += 1

        total_time += float(result["inference_time_ms"])

        status = "✅" if is_correct else "❌"
        print(f"{status} Query: {query}")
        print(f"   Expected: {expected_category}")
        print(f"   Got: {result['category']}")
        print(f"   Time: {result['inference_time_ms']}ms\n")

    # Summary
    accuracy = (correct / len(test_queries)) * 100
    avg_time = total_time / len(test_queries)

    print("="*60)
    print(f"📈 Results Summary:")
    print(f"   Accuracy: {accuracy:.1f}% ({correct}/{len(test_queries)})")
    print(f"   Average Inference Time: {avg_time:.2f}ms")
    print(f"   Total Inference Time: {total_time:.2f}ms")
    print("="*60)

    return accuracy, avg_time


def interactive_classify():
    """Interactive classification mode"""
    print("\n🚀 Starting interactive classifier...")
    print("Enter healthcare queries to classify (type 'quit' to exit)")
    print("="*60)

    classifier = QueryClassifier(use_quantization=True)

    while True:
        query = input("\n📝 Enter query: ").strip()

        if query.lower() == "quit":
            print("👋 Goodbye!")
            break

        if not query:
            continue

        result = classifier.classify(query)

        print(f"\n✨ Classification Result:")
        print(f"   Category: {result['category']}")
        print(f"   Confidence: {result['confidence']}")
        print(f"   Inference Time: {result['inference_time_ms']}ms")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        benchmark_classifier()
    elif len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_classify()
    else:
        print(__doc__)
        print("\nUsage:")
        print("  python classify_query.py benchmark     - Run benchmark tests")
        print("  python classify_query.py interactive   - Interactive mode")
        print("  python classify_query.py <query>       - Classify single query")

        if len(sys.argv) > 1:
            # Single query mode
            query = " ".join(sys.argv[1:])
            classifier = QueryClassifier()
            result = classifier.classify(query)

            print(f"\n📊 Classification Result:")
            for key, value in result.items():
                print(f"   {key}: {value}")
