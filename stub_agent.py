# stub_agent.py — temporary, for UI testing only, no API calls
import time


def run_pipeline(user_query, progress_callback=None):
    if progress_callback:
        for msg in ["Iteration 1/4 — 0 qualified so far", "Searching Indian retail sites...", "Checking prices...", "Generating final report..."]:
            progress_callback(msg)
            time.sleep(0.5)

    has_budget_word = "budget" in user_query.lower()
    has_number = any(char.isdigit() for char in user_query)
    has_budget = has_budget_word or has_number

    usecase_words = ["for", "coding", "gaming", "ml",
                     "machine learning", "editing", "study", "office", "work"]
    has_usecase = any(word in user_query.lower() for word in usecase_words)

    if not has_budget:
        return {'status': 'clarify', 'question': "What's your budget for this?"}
    if not has_usecase:
        return {'status': 'clarify', 'question': "What will you mainly use this laptop for?"}

    fake_report = """## 1. Requirements Summary
- **Use case:** Coding
- **Budget:** ₹50,000

## 2. Top Recommendations
| # | Product | Price | Fit Score |
|---|---------|-------|-----------|
| 1 | Sample Laptop X | ₹48,000 | 8/10 |
| 2 | Sample Laptop Y | ₹52,000 | 7/10 |

## 5. Final Recommendation
**Sample Laptop X** is the best fit within budget."""

    return {
        'status': 'done',
        'report': fake_report,
        'candidates': [],
        'is_degraded': False
    }
