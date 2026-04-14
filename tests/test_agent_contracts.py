import unittest
from pathlib import Path

from src.services.github_catalog import extract_repository_items
from src.services.portfolio_catalog import load_private_portfolio_catalog


ROOT = Path(__file__).resolve().parent.parent


class AgentContractTests(unittest.TestCase):
    def test_assistant_prompt_enforces_public_repo_followups(self):
        source = (ROOT / "src/agents/assistant.py").read_text(encoding="utf-8")
        self.assertIn("keep handing off related follow-up questions", source)
        self.assertIn("public_repo_catalog", source)
        self.assertIn("Do not generate code examples from search_portfolio results", source)

    def test_specialist_prompt_requires_repo_selection_before_mindmap(self):
        source = (ROOT / "src/agents/specialist.py").read_text(encoding="utf-8")
        self.assertIn("First determine which public repository", source)
        self.assertIn("Do not call mindmap.overview until the target repository is clear", source)
        self.assertIn("selected_repo_id", source)
        self.assertIn("owner/repo format", source)
        self.assertIn("If multiple repositories remain plausible", source)

    def test_private_catalog_uses_frontmatter_metadata(self):
        catalog = load_private_portfolio_catalog(str(ROOT / "data/raw"))
        names = {entry["project_name"] for entry in catalog}
        self.assertIn("Closync", names)
        self.assertIn("DentalPrices Virtual POS", names)
        self.assertTrue(all(entry["visibility"] == "private" for entry in catalog))
        self.assertTrue(any(entry["company_name"] == "Closync" for entry in catalog))

    def test_repository_items_parse_from_mcp_output_blocks(self):
        raw_result = {
            "output": [
                {
                    "type": "text",
                    "text": (
                        '{"total_count":2,"items":['
                        '{"name":"agentfolio","full_name":"engincankaya/agentfolio","language":"Python","updated_at":"2026-04-13T02:22:44Z"},'
                        '{"name":"claude-code-mind-map","full_name":"engincankaya/claude-code-mind-map","language":"TypeScript","updated_at":"2026-04-11T19:46:45Z"}'
                        "]}"
                    ),
                }
            ]
        }
        items = extract_repository_items(raw_result)
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["full_name"], "engincankaya/agentfolio")
        self.assertEqual(items[1]["language"], "TypeScript")

    def test_repository_items_parse_from_mcp_block_list(self):
        raw_result = [
            {
                "id": "lc_1",
                "type": "text",
                "text": (
                    '{"total_count":2,"items":['
                    '{"name":"agentfolio","full_name":"engincankaya/agentfolio","language":"Python","updated_at":"2026-04-13T02:22:44Z"},'
                    '{"name":"claude-code-mind-map","full_name":"engincankaya/claude-code-mind-map","language":"TypeScript","updated_at":"2026-04-11T19:46:45Z"}'
                    "]}"
                ),
            }
        ]
        items = extract_repository_items(raw_result)
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["name"], "agentfolio")


if __name__ == "__main__":
    unittest.main()
