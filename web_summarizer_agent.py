"""
Web Page Summarizer Agent
-------------------------
An agent that takes a URL, scrapes the webpage content, and summarizes it using an LLM.

Run with:
    uv run python core/examples/web_summarizer_agent.py
"""

import asyncio

from framework.graph import EdgeCondition, EdgeSpec, Goal, GraphSpec, NodeSpec
from framework.graph.executor import GraphExecutor
from framework.graph.node import NodeContext, NodeProtocol, NodeResult
from framework.llm.anthropic import AnthropicProvider
from framework.runtime.core import Runtime
from pathlib import Path


# 1. Web Scraping Node
class WebScraperNode(NodeProtocol):
    """Scrape content from a URL using Playwright for JS-rendered pages."""

    async def execute(self, ctx: NodeContext) -> NodeResult:
        url = ctx.input_data.get("url")
        if not url:
            return NodeResult(success=False, error="URL is required")

        # Check if it's a dynamic site (Spotify, etc.)
        dynamic_sites = ["spotify.com", "twitter.com", "x.com", "instagram.com"]

        try:
            if any(site in url.lower() for site in dynamic_sites):
                return await self._scrape_with_playwright(ctx, url)
            else:
                return await self._scrape_with_httpx(ctx, url)

        except Exception as e:
            return NodeResult(success=False, error=f"Scraping error: {e}")

    async def _scrape_with_playwright(self, ctx: NodeContext, url: str) -> NodeResult:
        """Use Playwright for JavaScript-rendered pages."""
        try:
            from playwright.async_api import async_playwright
            from bs4 import BeautifulSoup
        except ImportError:
            return NodeResult(success=False, error="Playwright not installed. Run: uv add playwright && playwright install chromium")

        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
                )

                await page.goto(url, wait_until="domcontentloaded", timeout=60000)
                await page.wait_for_timeout(3000)  # Wait for JS to render

                html = await page.content()
                await browser.close()

            return self._parse_html(ctx, url, html)

        except Exception as e:
            return NodeResult(success=False, error=f"Playwright error: {e}")

    async def _scrape_with_httpx(self, ctx: NodeContext, url: str) -> NodeResult:
        """Use httpx for static pages."""
        import httpx
        from bs4 import BeautifulSoup
        import urllib3
        import urllib3

        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        }

        transport = httpx.AsyncHTTPTransport(verify=False)
        async with httpx.AsyncClient(transport=transport, follow_redirects=True, timeout=30.0) as client:
            response = await client.get(url, headers=headers)

        if response.status_code != 200:
            return NodeResult(success=False, error=f"HTTP {response.status_code}")

        return self._parse_html(ctx, url, response.text)

    def _parse_html(self, ctx: NodeContext, url: str, html: str) -> NodeResult:
        """Parse HTML and extract content."""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")

        # Remove noise elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "iframe"]):
            tag.decompose()

        # Get title and description
        title = soup.title.get_text(strip=True) if soup.title else ""
        meta_desc = soup.find("meta", attrs={"name": "description"})
        description = meta_desc.get("content", "") if meta_desc else ""

        # Get main content
        main_content = (
            soup.find("article")
            or soup.find("main")
            or soup.find(attrs={"role": "main"})
            or soup.find(class_=["content", "post", "entry", "article-body"])
            or soup.find("body")
        )
        content = main_content.get_text(separator=" ", strip=True) if main_content else ""

        # Clean up whitespace
        content = " ".join(content.split())

        # Limit content length
        max_length = 50000
        if len(content) > max_length:
            content = content[:max_length] + "..."

        if not content.strip():
            return NodeResult(success=False, error="No content extracted - page may require authentication or has dynamic content")

        # Store scraped content in memory (validate=False for large content)
        ctx.memory.write("scraped_title", title, validate=False)
        ctx.memory.write("scraped_description", description, validate=False)
        ctx.memory.write("scraped_content", content, validate=False)
        ctx.memory.write("scraped_url", url, validate=False)

        return NodeResult(
            success=True,
            output={
                "title": title,
                "content": content,
                "url": url,
                "full_content": content,
            },
        )


# 2. Summarization Node
class SummarizerNode(NodeProtocol):
    """Summarize the scraped content using an LLM."""

    async def execute(self, ctx: NodeContext) -> NodeResult:
        content = ctx.memory.read("scraped_content")
        title = ctx.memory.read("scraped_title")
        url = ctx.memory.read("scraped_url")
        api_key = ctx.input_data.get("api_key") or ctx.memory.read("api_key")

        if not content:
            return NodeResult(success=False, error="No content to summarize")

        try:
            # Check if API key is available
            if not api_key:
                import os
                api_key = os.environ.get("ANTHROPIC_API_KEY")

            if not api_key:
                # Fallback: Use a simple extractive summary if no API key
                summary = self._extractive_summary(content, title)
            else:
                # Use LLM for summarization
                llm = AnthropicProvider(api_key=api_key)

                system_prompt = """You are a helpful assistant that summarizes web content.
Provide a clear, concise summary of the content below. Include:
- The main topic or purpose
- Key points or information
- Any important conclusions or takeaways

Keep the summary informative but easy to understand."""

                user_message = f"""Title: {title}

URL: {url}

Content to summarize:
{content}

Please provide a comprehensive summary:"""

                response = llm.acomplete(
                    messages=[
                        {"role": "user", "content": user_message}
                    ],
                    system=system_prompt,
                    max_tokens=2048,
                )

                summary = response.content

            # Store summary in memory
            ctx.memory.write("summary", summary)

            return NodeResult(
                success=True,
                output={
                    "summary": summary,
                    "url": url,
                    "title": title,
                },
            )

        except Exception as e:
            # Fallback to extractive summary on error
            summary = self._extractive_summary(content, title)
            ctx.memory.write("summary", summary)
            return NodeResult(
                success=True,
                output={
                    "summary": summary,
                    "url": url,
                    "title": title,
                },
            )

    def _extractive_summary(self, content: str, title: str) -> str:
        """Generate a simple extractive summary when LLM is not available."""
        # Take first few sentences as summary
        sentences = content.split(". ")
        summary_parts = [title] if title else []
        for sent in sentences[:5]:
            if len(sent) > 30:  # Skip very short sentences
                summary_parts.append(sent if sent.endswith(".") else sent + ".")
        return " ".join(summary_parts)


async def main():
    print("Web Page Summarizer Agent")
    print("=" * 40)

    # Get URL from user
    url = input("Enter a URL to summarize: ").strip()

    if not url:
        print("Error: URL is required")
        return

    # Add https:// if missing
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    print(f"\nFetching and summarizing: {url}")
    print("-" * 40)

    # Define the Goal
    goal = Goal(
        id="summarize-webpage",
        name="Summarize Webpage",
        description="Scrape a webpage and generate a summary using AI",
        success_criteria=[
            {
                "id": "summary_generated",
                "description": "Summary produced successfully",
                "metric": "custom",
                "target": "any",
            }
        ],
    )

    # Define Nodes
    scraper_node = NodeSpec(
        id="scraper",
        name="Web Scraper",
        description="Scrape content from the URL",
        node_type="event_loop",
        input_keys=["url"],
        output_keys=["scraped_title", "scraped_description", "scraped_content", "scraped_url", "title", "content"],
    )

    summarizer_node = NodeSpec(
        id="summarizer",
        name="Content Summarizer",
        description="Generate a summary using AI",
        node_type="event_loop",
        input_keys=["scraped_title", "scraped_description", "scraped_content", "scraped_url", "api_key"],
        output_keys=["summary"],
    )

    # Define Edges
    edge = EdgeSpec(
        id="scrape-to-summarize",
        source="scraper",
        target="summarizer",
        condition=EdgeCondition.ON_SUCCESS,
    )

    # Create Graph
    graph = GraphSpec(
        id="web-summarizer",
        goal_id="summarize-webpage",
        entry_node="scraper",
        terminal_nodes=["summarizer"],
        nodes=[scraper_node, summarizer_node],
        edges=[edge],
    )

    # Initialize Runtime & Executor
    runtime = Runtime(storage_path=Path("./agent_logs"))
    executor = GraphExecutor(runtime=runtime)

    # Register Node Implementations
    executor.register_node("scraper", WebScraperNode())
    executor.register_node("summarizer", SummarizerNode())

    # Execute Agent
    result = await executor.execute(graph=graph, goal=goal, input_data={"url": url})

    # Display Results
    if result.success:
        print("\n" + "=" * 40)
        print("SUCCESS!")
        print("=" * 40)
        print(f"\nTitle: {result.output.get('title', 'N/A')}")
        print(f"\nSummary:\n{result.output.get('summary', 'N/A')}")
        # Show full content for debugging
        print(f"\n--- Full Content Preview (first 3000 chars) ---")
        print(result.output.get('full_content', 'N/A')[:3000])
    else:
        print(f"\nError: {result.error}")


if __name__ == "__main__":
    asyncio.run(main())