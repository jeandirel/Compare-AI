AI Model Performance Dashboard
Welcome to the comparative analysis dashboard for AI models. This tool is designed to explore and visualize the trade-offs between answer quality, environmental impact (CO2 emissions), energy cost (Wh), and speed (latency) across various AI models.

How to Use the Dashboard
This dashboard is interactive. Use the filters and explore the different sections to refine your analysis and derive strategic insights.

1. Using the Sidebar Filters
The sidebar on the left of the screen contains the primary controls for customizing the data display.

A. Filter by Model Size (Type)
Function: Allows you to select or deselect models based on their size: Small, Medium, or Large.

Usage: Uncheck a category to remove it from all visualizations. This is useful for comparing only models of a certain scale.

B. Filter by Prompt Category
Function: Allows you to segment the analysis based on the type of task given to the model.

Options:

All: Displays data for all tasks (default setting).

Literary/General: Tasks involving writing, summarization, translation, etc.

Mathematical/Logical: Tasks involving calculations and logical problem-solving.

Coding: Tasks related to code generation or debugging.

Usage: Select a specific category to evaluate how models perform on that precise type of task.

2. Understanding the Analysis Sections
The dashboard is divided into several sections for a progressive analysis, from a general overview to fine-grained details.

Section I: Overall Performance Analysis
This section provides a high-level view of model performance.

Key Performance Indicators (KPIs): Four metrics at the top of the page highlighting the single best model for each key metric (Highest Quality, Lowest CO2, etc.).

Comparative Metric Analysis: A grid of four bar charts that compares all filtered models on each individual metric. This is ideal for quickly seeing the leader and laggards on a specific criterion.

Core Performance Trade-offs: Two bubble charts that reveal the compromises:

Quality vs. CO2 Emission: The ideal model is in the top-left corner (high quality, low CO2).

Quality vs. Cost: The ideal model is also in the top-left corner (high quality, low cost).

How to read the bubbles:

Color: Identifies the model.

Shape (circle, cross, diamond): Indicates the model's size (Type).

Bubble Size: Represents a third dimension (either cost or CO2).

Section II: Deep Dive & Data
This section is accessible via tabs and allows for a more detailed analysis.

Tab "Performance by Category": This view is only available when the "All" prompt filter is selected. It compares model performance (quality and cost) across the different prompt categories (Literary, Math, Coding) side-by-side. This is where you can investigate if a "Large" model is genuinely better at coding than a "Small" one.

Tab "Detailed Statistics": Displays a summary table with the average performance figures for each model across the four primary metrics.

Tab "Raw Data Viewer": Provides access to the underlying raw data used to generate the charts, reflecting any filters you have applied.

Metric Definitions
Quality (1-5): The quality of the model's response, rated on a scale of 5. A higher score is better.

CO2 Emission (g): The equivalent grams of CO2 emitted to complete the task. A lower number is better.

Cost (Wh): The energy cost in Watt-hours consumed for the task. A lower number is better.

Latency (s): The time in seconds required to receive a response. A lower number is better.
