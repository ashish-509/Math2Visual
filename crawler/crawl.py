import asyncio
from crawl4ai import AsyncWebCrawler


async def main():
    # Create an instance of AsyncWebCrawler
    async with AsyncWebCrawler() as crawler:
        # Run the crawler on Manim documentation URL
        result = await crawler.arun(url="https://docs.manim.community/en/stable/index.html")
        
        # Save the extracted content to a markdown file
        with open("markdown_output.md", "w", encoding="utf-8") as f:
            f.write(result.markdown)
        
        # Print the extracted content for verification
        print(result.markdown)

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())

