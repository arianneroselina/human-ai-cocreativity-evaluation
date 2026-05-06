# Human-AI Co-Creativity Evaluation 🧠📊

[![Vercel](https://img.shields.io/badge/Deploy%20on-Vercel-000000?logo=vercel)](https://vercel.com)
[![Prisma](https://img.shields.io/badge/Powered%20by-Prisma-2D3748?logo=prisma)](https://www.prisma.io)
[![Node.js](https://img.shields.io/badge/Node.js-v16.13.0-339933?logo=node.js)](https://nodejs.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-13.2-336791?logo=postgresql)](https://www.postgresql.org/)
[![GitHub](https://img.shields.io/badge/Source%20Code-GitHub-181717?logo=github)](https://github.com)

A platform designed to evaluate creative outputs from a human–AI co-creativity study through anonymous poem rating.

The system enables unbiased assessment of poems written by humans, AI, or human–AI collaborations. Evaluators are shown one poem at a time and rate it across several evaluation metrics on a scale from 1 to 10.

---

## 🚀 Demo - Try it Now!

You can explore the evaluation interface by running the project locally:

👉 http://localhost:3000

## 📝 Evaluation Design

The evaluation is based on a controlled dataset where:

- 24 participants each wrote poems across 7 different topics
- Each poem was created under one of the following conditions:
    - **Human-only**
    - **AI-only**
    - **Human→AI**
    - **AI→Human**

The original dataset contains **168 poems**. Since 2 poems are empty, the final evaluation dataset consists of **166 valid poems**.

## ⚖️ Evaluation Method

The platform uses a blind single-poem rating approach:

- One poem is shown at a time
- The poem topic is displayed for context
- Authorship and creation condition are hidden
- Evaluators rate each poem across different metrics
- Each metric is rated on a scale from **1 to 10**

This approach reduces authorship bias and allows each poem to be evaluated independently based on its creative quality.

## 🛠️ Project Setup

### 1. **Install Dependencies**

Install the necessary dependencies:

```bash
npm install
```

### 2. **Start Development**

To start the development server locally, run:

```bash
make run
```

Then navigate to [http://localhost:3000](http://localhost:3000) in your browser to start interacting with the platform.

## 💡 Acknowledgements

This project is part of a Master Thesis exploring human–AI co-creativity and the evaluation of generative outputs.
Special thanks to all participants contributing to the dataset.