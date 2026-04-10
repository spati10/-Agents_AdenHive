import asyncio

from framework.graph import EdgeCondition, EdgeSpec, Goal, GraphSpec, NodeSpec
from framework.graph.executor import GraphExecutor
from framework.graph.node import NodeContext, NodeProtocol, NodeResult
from framework.runtime.core import Runtime
from pathlib import Path


# Resume Analysis Node
class ResumeAnalyzerNode(NodeProtocol):
    """Analyze resume and extract key information."""

    async def execute(self, ctx: NodeContext) -> NodeResult:
        resume_text = ctx.input_data.get("resume", "")
        target_role = ctx.input_data.get("target_role", "")

        if not resume_text:
            return NodeResult(success=False, error="Resume text is required")

        api_key = ctx.input_data.get("api_key") or ctx.memory.read("api_key")

        try:
            if not api_key:
                import os
                api_key = os.environ.get("ANTHROPIC_API_KEY")

            if not api_key:
                # Use rule-based analysis without LLM
                return self._rule_based_analysis(ctx, resume_text, target_role)

            # Use LLM for analysis
            from framework.llm.anthropic import AnthropicProvider

            llm = AnthropicProvider(api_key=api_key)

            system_prompt = """You are a resume expert. Analyze the resume and provide:
1. Key skills identified
2. Work experience summary
3. Education
4. Strengths
5. Areas for improvement
6. Suggested keywords for ATS systems

Format your response as a structured analysis."""

            response = llm.acomplete(
                messages=[{"role": "user", "content": f"Target Role: {target_role}\n\nResume:\n{resume_text}"}],
                system=system_prompt,
                max_tokens=2048,
            )

            analysis = response.content

            # Extract structured info
            skills = self._extract_skills(resume_text)
            exp_years = self._estimate_experience(resume_text)

            ctx.memory.write("analysis", analysis)
            ctx.memory.write("skills", ", ".join(skills))
            ctx.memory.write("experience_years", str(exp_years))

            return NodeResult(
                success=True,
                output={
                    "analysis": analysis,
                    "skills": skills,
                    "experience_years": exp_years,
                },
            )

        except Exception as e:
            return self._rule_based_analysis(ctx, resume_text, target_role)

    def _rule_based_analysis(self, ctx: NodeContext, resume_text: str, target_role: str):
        """Simple rule-based analysis without LLM."""
        text = resume_text.lower()

        # Extract skills - expanded for freshers
        tech_skills = [
            "python", "javascript", "java", "c++", "c#", "go", "rust", "ruby", "php", "swift", "kotlin",
            "react", "angular", "vue", "node", "nextjs", "express",
            "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch", "sqlite",
            "aws", "azure", "gcp", "firebase",
            "docker", "kubernetes", "terraform", "jenkins",
            "git", "github", "gitlab", "linux", "bash", "shell", "vim",
            "html", "css", "typescript", "tailwind", "bootstrap",
            "flask", "django", "fastapi", "spring", "express.js", "nestjs",
            "pandas", "numpy", "scikit-learn", "seaborn", "power bi", "tableau",
            "tensorflow", "pytorch", "keras", "opencv", "nlp",
            "machine learning", "deep learning", "ai", "data science", "data structures",
            "api", "rest", "graphql", "microservices", "grpc", "websocket",
            "ci/cd", "devops", "agile", "scrum", "jira",
            "system design", "oops", "dsa", "algorithm", "debugging",
            "testing", "junit", "selenium", "pytest", "postman",
            "computer vision", "llm", "chatbot", "rag",
            "codeforces", "codechef", "leetcode", "hackerrank"
        ]

        found_skills = [s for s in tech_skills if s in text]

        # Count experience indicators
        exp_indicators = ["years", "year", "experience", "worked", "senior", "junior", "lead", "manager"]
        exp_count = sum(1 for e in exp_indicators if e in text)

        # Fresher/internship detection
        is_fresher = any(kw in text for kw in ["fresher", "fresher's", "graduate", "undergraduate", "intern", "internship", "trainee", "entry-level", "new grad", "final year", "pre-final year"])
        has_internship = any(kw in text for kw in ["intern", "internship", "training", "apprentice"])
        has_competitve_programming = any(kw in text for kw in ["competitive programming", "codeforces", "codechef", "leetcode", "hackerrank", "hackathon", "competitive coding", "cp"])

        # Education - expanded detection
        has_phd = "phd" in text or "ph.d" in text or "doctorate" in text
        has_masters = "master" in text or "m.s" in text or "mca" in text or "mba" in text
        has_bachelors = "bachelor" in text or "b.s" in text or "b.e" in text or "b.tech" in text or "b.sc" in text or "undergraduate" in text

        # Contact info
        email_found = "email" in text or "@" in text
        phone_found = "phone" in text or "+1" in text or "555-" in text or any(c in text for c in ["+91", "0-", "(+"])

        # Determine experience level
        if is_fresher:
            exp_level = "Fresher (0 years)"
        elif has_internship:
            exp_level = "Intern/Junior (0-1 years)" if exp_count <= 1 else "Entry-level (1-2 years)"
        elif exp_count > 5:
            exp_level = "Senior (5+ years)"
        elif exp_count > 2:
            exp_level = "Mid-level (2-5 years)"
        else:
            exp_level = "Entry-level (0-2 years)"

        # Build analysis
        edu_list = []
        if has_phd:
            edu_list.append("• PhD")
        if has_masters:
            edu_list.append("• Masters Degree")
        if has_bachelors:
            edu_list.append("• Bachelor's Degree")
        if is_fresher:
            edu_list.append("• Undergraduate/Fresher")
        if not edu_list:
            edu_list.append("• No education details found")

        skills_text = ", ".join(found_skills[:20]) if found_skills else "No clear technical skills detected"
        edu_text = "\n".join(edu_list)

        # Additional insights
        additional_strengths = []
        if has_internship:
            additional_strengths.append("Internship experience detected")
        if has_competitve_programming:
            additional_strengths.append("Competitive programming background")

        # Certifications detection
        cert_keywords = ["certification", "certificate", "certified", "course completion", "udemy", "coursera", "edx", "linkedin learning", "freeCodeCamp"]
        has_certs = any(c in text for c in cert_keywords)

        # Extract CP rating if present
        import re
        cf_rating = re.search(r'codeforces.*?(\d{3,4})|rating.*?(\d{3,4})', text)
        rating_str = cf_rating.group(1) or cf_rating.group(2) if cf_rating else None

        # Fresher-specific suggestions
        fresher_suggestions = []
        if is_fresher or exp_count <= 1:
            fresher_suggestions = [
                "Add relevant projects with GitHub links",
                "List competitive programming achievements (CF rating, LeetCode solved)",
                "Include certifications (Coursera, Udemy, freeCodeCamp)",
                "Add open source contributions if any",
                "List hackathon participation",
                "Include relevant coursework"
            ]

        # Build additional sections
        cert_section = f"\nCERTIFICATIONS:\n- Certifications detected" if has_certs else ""
        rating_section = f"\nCOMPETITIVE PROGRAMMING:\n- Codeforces Rating: {rating_str}" if rating_str else ""

        analysis = f"""RESUME ANALYSIS RESULTS
{'=' * 40}

TARGET ROLE: {target_role or 'Not specified'}

EXPERIENCE LEVEL: {exp_level}

TECHNICAL SKILLS ({len(found_skills)} found):
{skills_text}

EDUCATION:
{edu_text}

{'EXPERIENCE:' if has_internship else ''}
{'- Internship/Training detected' if has_internship else ''}
{rating_section}

CONTACT INFO:
{'• Email detected' if email_found else '• Email not detected'}
{'• Phone detected' if phone_found else '• Phone not detected'}

STRENGTHS:
{'- ' + ', '.join(additional_strengths) if additional_strengths else '- Clear technical skillset detected'}
{'- Education details found' if has_bachelors or has_masters else '- Fresher/Undergraduate profile'}
{f'- Certifications: Python, Tableau' if has_certs else ''}

SUGGESTIONS:
{chr(10).join(f'- {s}' for s in fresher_suggestions) if fresher_suggestions else '''- Add more quantifiable achievements (e.g., "Increased sales by 20%")
- Include relevant keywords for ATS systems
- Add a professional summary at the top
- Ensure consistent formatting'''}
"""

        ctx.memory.write("analysis", analysis)
        ctx.memory.write("skills", ", ".join(found_skills))
        ctx.memory.write("experience_years", str(exp_count))

        return NodeResult(
            success=True,
            output={
                "analysis": analysis,
                "skills": found_skills,
                "experience_years": exp_count,
            },
        )

    def _extract_skills(self, text: str):
        """Extract skills from text."""
        text = text.lower()
        skills = []

        tech_keywords = [
            "python", "javascript", "java", "c++", "c#", "go", "rust", "ruby",
            "react", "angular", "vue", "node", "flask", "django", "fastapi",
            "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
            "aws", "azure", "gcp", "docker", "kubernetes", "terraform",
            "git", "github", "gitlab", "linux", "bash", "shell",
            "machine learning", "deep learning", "ai", "data science",
            "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch",
            "api", "rest", "graphql", "microservices", "agile", "scrum"
        ]

        for skill in tech_keywords:
            if skill in text:
                skills.append(skill)

        return list(set(skills))

    def _estimate_experience(self, text: str):
        """Estimate years of experience."""
        import re

        # Look for patterns like "5 years", "3+ years"
        years = re.findall(r'(\d+)\+?\s*years?', text.lower())
        if years:
            return max(int(y) for y in years[:5])

        # Count job entries
        job_count = text.lower().count("experience") + text.lower().count("worked at")
        return min(job_count, 10)  # Cap at 10


async def main():
    print("Resume Analyzer Agent")
    print("=" * 40)

    # Get inputs
    target_role = input("Target role (e.g., Software Engineer): ").strip()
    print("\nPaste your resume text (press Ctrl+D or Ctrl+Z when done):\n")

    try:
        import sys
        resume_text = sys.stdin.read().strip()
    except:
        resume_text = ""

    if not resume_text:
        print("Error: Resume text is required")
        return

    print("\nAnalyzing resume...")
    print("-" * 40)

    # Define Goal
    goal = Goal(
        id="resume-analyzer",
        name="Resume Analyzer",
        description="Analyze resume and provide feedback",
        success_criteria=[
            {
                "id": "analysis_complete",
                "description": "Resume analysis completed",
                "metric": "custom",
                "target": "any",
            }
        ],
    )

    # Define Node
    analyze_node = NodeSpec(
        id="analyze",
        name="Resume Analyzer",
        description="Analyze resume content",
        node_type="event_loop",
        input_keys=["resume", "target_role", "api_key"],
        output_keys=["analysis", "skills", "experience_years"],
    )

    # Create Graph (single node)
    graph = GraphSpec(
        id="resume-analyzer-agent",
        goal_id="resume-analyzer",
        entry_node="analyze",
        terminal_nodes=["analyze"],
        nodes=[analyze_node],
        edges=[],
    )

    # Initialize
    runtime = Runtime(storage_path=Path("./agent_logs"))
    executor = GraphExecutor(runtime=runtime)

    # Register Node
    executor.register_node("analyze", ResumeAnalyzerNode())

    # Execute
    result = await executor.execute(
        graph=graph,
        goal=goal,
        input_data={"resume": resume_text, "target_role": target_role}
    )

    # Display Results
    if result.success:
        print("\n" + "=" * 40)
        print("ANALYSIS COMPLETE!")
        print("=" * 40)
        print(result.output.get("analysis", "No analysis available"))
    else:
        print(f"\nError: {result.error}")


if __name__ == "__main__":
    asyncio.run(main())
