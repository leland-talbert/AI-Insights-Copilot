## AI Insights Copilot (Capability Demo)

This repository contains a fictional, demonstration-only prototype exploring
how Generative AI can support enterprise decision-making with:

- Audience-specific insights (Executive, Finance, Technology, Analyst)
- Human-in-the-loop review by design
- Safety, governance, and cost awareness

📽 **Live Demo Walkthrough (2 min):**
https://www.linkedin.com/feed/update/urn:li:activity:7406702198795153408/

⚠️ All data and content are fictional and generated solely for demonstration purposes.
No proprietary, confidential, or employer-related information was used.

This demo reflects how I approach real-world GenAI system design: 
business-first, safety-aware, and human-reviewed by default.


## Architecture & Stack

- Python 3.13
- Streamlit
- OpenAI / LLM inference
- Role-aware prompt orchestration
- Human-in-the-loop validation
  
<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.13-blue" />
  <img alt="Streamlit" src="https://img.shields.io/badge/streamlit-app-ff4b4b" />
</p>
  <!-- Replace with your real links -->



---

## Overview
AI Insights Copilot is a Streamlit-based Generative AI assistant that converts natural-language questions into:
- **Structured summaries**
- **Key takeaways**
- **Tables / analytics**
- **Visualization-ready outputs**

It’s designed as an enterprise-ready prototype: modular app architecture, secure configuration via environment variables, and a clear path to cloud deployment.

---

## Demo

![AI Insights Copilot Dashboard](docs/images/dashboard.png)


---

## Key Features
- **Natural-language query input** → insight generation
- **Prompting + routing** to shape outputs based on intent
- **LLM API integration** (secure via `.env`)
- **Dashboard UX** built for recruiter/business workflows
- **Extensible data layer** (CSV/JSON now; SQL/APIs/RAG later)

---

## Architecture

![Architecture Diagram](docs/images/architecture.png)


**High-level flow**
1. Streamlit UI captures query  
2. App layer parses intent + builds prompt  
3. LLM inference generates response  
4. Output is formatted into summaries + visuals

---

## Tech Stack
- **Python 3.13**
- **Streamlit**
- **LLM API** (OpenAI / Gemini / etc. — pluggable)
- **Pandas** for data shaping
- **Matplotlib/Plotly** (optional) for charts

---

## Getting Started

### 1) Clone the repo
```bash
git clone https://github.com/leland-talbert/ai-insights-copilot.git
cd ai-insights-copilot


