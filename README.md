# AI-Powered Cover Letter Generator

An intelligent cover letter generator that uses LLM agents, semantic search, and skill matching to create personalized, professional cover letters tailored to specific job positions.

## Features

- **CV Parsing**: Automatically extracts structured information from .docx CVs
- **Job Analysis**: Parses job descriptions to understand requirements and skills
- **Skill Matching**: Analyzes the match between your CV and job requirements with semantic similarity
- **Semantic Search**: Uses FAISS vector store for intelligent CV content retrieval
- **Multiple Tones**: Generates cover letters in three distinct styles:
  - **Professional**: Formal, achievement-focused, traditional business format
  - **Creative**: Engaging narrative, shows personality while remaining professional
  - **Technical**: Detailed technical expertise, specific technologies and projects
- **Interactive Refinement**: Chat interface to refine and improve generated letters
- **Factually Grounded**: All generated content is based strictly on information from your CV—no invented details or assumptions
- **Powered by LangGraph**: Uses ReAct agents for intelligent, tool-based generation

## Architecture

```
CV (.docx) + Job Description (.txt)
  ↓
[Parse & Structure Documents]
  ↓
[Create FAISS Vector Store from CV]
  ↓
[Skill Matching Analysis]
  ↓
[LangGraph ReAct Agent with Tools]
  • search_cv (semantic search)
  • analyze_skill_match
  • get_job_details
  • get_cv_summary
  • generate_cover_letter_content
  ↓
[Generate 3 Versions: Professional, Creative, Technical]
  ↓
[Interactive Chat for Refinement]
  ↓
Save Final Cover Letters
```

## Installation

### Prerequisites

- Python 3.12+
- Virtual environment (venv)
- OpenAI API key

### Setup

1. **Activate your virtual environment**:
```bash
cd "/Users/bitarahmatzade/Downloads/cv bita/personalProjects/LangChain"
source venv/bin/activate
```

2. **Dependencies are already installed**:
- langchain, langchain-openai, langgraph
- faiss-cpu for vector search
- python-docx for CV parsing
- All other required packages

3. **Ensure OpenAI API key is set** in your `.env` file:
```bash
OPENAI_API_KEY=sk-your-key-here
```

## Usage

### Basic Usage

1. **Prepare your inputs**:
   - Place your CV (.docx) in `data/cv/` or have it ready
   - Place job description (.txt) in `data/jobs/` or have it ready

2. **Run the generator**:
```bash
python main_cover_letter.py
```

3. **Follow the prompts**:
   - Enter path to your CV (.docx file)
   - Enter path to job description (.txt file)
   - The system will generate 3 versions and save them
   - You can then refine any version interactively

### Interactive Refinement Commands

Once in refinement mode, you can:
- **Request changes**: "Make it more concise", "Add more technical details"
- **Emphasize content**: "Focus more on my AWS experience"
- **Change tone**: "Make it sound more enthusiastic"
- **Fix issues**: "Remove the mention of X", "Update the opening paragraph"
- **Show current**: Type `show` to see the current version
- **Exit**: Type `done`, `exit`, or `quit` to finish

## Project Structure

```
LangChain/
├── cover_letter_generator/           # Main package
│   ├── models.py                     # Pydantic data models
│   ├── config.py                     # Configuration
│   ├── parsers/
│   │   ├── cv_parser.py              # CV .docx parser
│   │   └── job_parser.py             # Job description parser
│   ├── analysis/
│   │   ├── skill_extractor.py        # LLM-based skill extraction
│   │   └── skill_matcher.py          # Semantic skill matching
│   ├── vector_store/
│   │   └── cv_vectorstore.py         # FAISS embeddings & retrieval
│   ├── agent/
│   │   ├── tools.py                  # Agent tools
│   │   ├── prompts.py                # Prompt management
│   │   └── cover_letter_agent.py     # LangGraph agent
│   └── utils/
│       └── validation.py             # Input validation
├── data/
│   ├── cv/                           # Your CV files (.docx)
│   └── jobs/                         # Job descriptions (.txt)
├── output/
│   └── cover_letters/                # Generated letters (timestamped)
├── prompts/
│   ├── professional_tone.txt         # Professional tone prompt
│   ├── creative_tone.txt             # Creative tone prompt
│   └── technical_tone.txt            # Technical tone prompt
├── main_cover_letter.py              # Main CLI application
├── interactive_refinement.py         # Chat refinement interface
└── README_COVER_LETTER.md            # This file
```

## How It Works

### 1. Document Parsing
- **CV Parser**: Extracts text from .docx, identifies sections (Experience, Education, Skills, Projects), and structures data
- **Job Parser**: Parses job description, extracts requirements, skills, and responsibilities

### 2. Semantic Indexing
- Creates FAISS vector store with embeddings of each CV section
- Enables semantic search to find most relevant experiences for the job

### 3. Skill Analysis
- Extracts skills from both CV and job description using LLM
- Performs exact and semantic matching
- Calculates match percentage and identifies gaps

### 4. Agent-Based Generation
- LangGraph ReAct agent uses tools to:
  - Search CV for relevant content
  - Analyze skill matches
  - Understand job requirements
  - Generate tailored cover letter with specified tone

### 5. Interactive Refinement
- Chat interface with conversation memory
- Stream responses for real-time feedback
- Iteratively improve the letter based on your feedback

## Technical Details

### Models Used
- **Generation**: GPT-4o (high quality, creative writing)
- **Skill Extraction**: GPT-4o-mini (fast, cost-effective)
- **Embeddings**: text-embedding-3-large (semantic search)

### Key Technologies
- **LangChain**: LLM framework and tool integration
- **LangGraph**: ReAct agent with checkpointing
- **FAISS**: Vector similarity search
- **python-docx**: Document parsing
- **Pydantic**: Data validation and modeling

## Tips for Best Results

### CV Preparation
- Use a well-structured .docx file with clear section headers
- Include quantifiable achievements with metrics
- List skills clearly (comma-separated or bullet points)
- Include relevant projects with descriptions

### Job Description Preparation
- Save job posting as plain text (.txt file)
- Keep the original structure with section headers
- Include all requirements and qualifications
- Don't heavily edit or summarize

### Choosing a Tone
- **Professional**: Corporate jobs, formal industries (finance, law, consulting)
- **Creative**: Startups, design roles, marketing positions
- **Technical**: Engineering roles, data science, technical positions

### Refinement Tips
- Be specific in your requests ("Add metrics to the second paragraph")
- Review skill match analysis to understand gaps
- Use the search_cv tool results to guide emphasis
- Iterate 2-3 times for best results
- All generated content is based strictly on your CV—refinement requests will enhance how your real achievements are presented, not invent new ones
- The system will ignore requests to add information not in your CV

## Cost Estimation

Approximate OpenAI API costs per cover letter generation:
- CV Parsing & Embeddings: ~$0.01
- Skill Analysis: ~$0.005
- 3 Cover Letter Versions: ~$0.03-0.05
- **Total per run: ~$0.05-0.07**

Interactive refinement adds ~$0.01-0.02 per exchange.

## Troubleshooting

### "CV must contain at least one work experience"
- Ensure your CV has a clear "Experience" or "Work History" section
- Check that experience entries are properly formatted

### "Job description is too short"
- Job description must be at least 50 characters
- Include full job posting, not just title

### "OpenAI API key not found"
- Verify `.env` file contains `OPENAI_API_KEY=sk-...`
- Ensure you're in the correct directory with activated venv

### Poor Skill Matching
- Review extracted skills in the output
- Ensure CV skills section is clearly labeled
- Job description should explicitly list required skills

### Generated Letter Quality Issues
- Try different tones - technical roles benefit from technical tone
- Use interactive refinement to guide the agent
- Provide more structured CV sections with clear achievements and metrics
- Ensure job description has clear requirements
- Note: Quality depends on content in your CV. Include specific achievements, metrics, and details in your CV for better letter generation

## Future Enhancements

Potential improvements:
- Export to formatted .docx or PDF
- ATS (Applicant Tracking System) optimization scoring
- Multi-language support
- Web interface (Streamlit/Gradio)
- Batch processing for multiple jobs
- Cover letter templates library
- Email draft generation

## Contributing

This is a personal project, but suggestions and improvements are welcome!

## License

For personal use. Modify and extend as needed for your job applications.

---

Built with LangChain, LangGraph, and OpenAI GPT-4o.
