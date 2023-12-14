# Data Scientist NLP Apprenticeship Take-Home

## Introduction

Welcome to the Fetch Data Scientist NLP Apprenticeship Take-Home project! This assignment involves building a tool that empowers users to intelligently search for offers via text input. The goal is to enhance user experience and provide valuable recommendations for both users and Fetch partners.

## Problem Statement

Fetch provides value to its user base through a variety of active offers in the app. The objective is to enable users to easily seek out relevant offers using a text-based search. This requires building a tool that can intelligently understand user queries and recommend offers based on categories, brands, and retailers.

## Acceptance Criteria

- If a user searches for a category (e.g., diapers), the tool should return a list of offers relevant to that category.
- If a user searches for a brand (e.g., Huggies), the tool should return a list of offers relevant to that brand.
- If a user searches for a retailer (e.g., Target), the tool should return a list of offers relevant to that retailer.
- The tool should also return the score used to measure the similarity of the text input with each offer.

## Solution Overview

### Approach

In addressing the challenge, a multifaceted approach was taken, focusing on simplicity, efficiency, and user experience:

#### Assumptions

1. **Text Preprocessing:** Robust preprocessing, including lemmatization, stop-word removal, and tokenization, is crucial for accurate similarity measurements.

2. **Similarity Score:** A combination of basic text similarity and advanced TF-IDF is used for accuracy and computational efficiency.

3. **Keyword Extraction:** Extracting keywords enhances relevancy assessments using a straightforward CountVectorizer method.

4. **Duplicate Removal:** Eliminating duplicate offers with higher similarity scores improves user experience.

5. **User Interface:** Dash web application for clear and interactive user input and recommendation display.

#### Trade-offs

1. **Complexity vs. Performance:** Maintaining computational efficiency while balancing algorithmic complexity.

2. **Threshold for Duplicate Removal:** A trade-off between precision and recall in setting the duplicate removal threshold.

3. **Keyword Extraction Method:** CountVectorizer chosen for simplicity.

4. **User Interaction:** Intentional minimalistic design for ease of use.

### Prompt Engineering Consideration

#### Overview:

In the development of the recommendation tool, a key consideration was given to prompt engineering—a technique involving the construction of specialized queries or prompts to enhance the model's understanding of user queries. This section outlines the assumptions made, the benefits anticipated, and the trade-offs considered in implementing prompt engineering.

#### Assumption:

Semantic Understanding: The foundational assumption was that prompt engineering would significantly contribute to capturing the semantic meaning embedded in user queries. For instance, when a user searches for a specific brand like "Huggies," prompt engineering might aid in extracting related keywords such as "diapers," "kids," or other pertinent categories.

#### Trade-off and Decision:

Computational Intensity: Acknowledging the advantage of improved semantic understanding through prompt engineering, a conscious trade-off was made between computational complexity and semantic richness. Balancing model performance and computational efficiency became pivotal to meet project goals.

Time Constraints: Given the time constraints imposed by the interview process, there was a need to prioritize the current implementation. Emphasis was placed on achieving a practical balance between accuracy and speed to create a user-friendly tool within the given timeframe.

### Future Consideration

Aspiring to contribute further, if selected as an intern, I am eager to delve into the intricacies of prompt engineering. Exploring more computationally intensive implementations, such as leveraging advanced models like BERT, holds promise for nuanced semantic understanding and precise recommendations. I look forward to collaborating on refining and expanding the tool, aligning it with evolving requirements and pushing its capabilities to new heights.

## How to Run Locally

1. Clone the repository to your local machine:

    ```bash
    git clone https://github.com/Tarun-MR/Fetch
    ```

2. Navigate to the project directory:

    ```bash
    cd Fetch
    ```

3. Run the script:

    ```bash
    python final.py
    ```

5. Open your web browser and go to [http://127.0.0.1:9030/](http://127.0.0.1:9030/) to access the Offer Recommendation Dashboard.

## File Structure

- `final.py`: Main script implementing the offer recommendation dashboard.
- `offer_retailer.csv`: Dataset containing offers and associated metadata.
- `brand_category.csv`: Dataset containing supported brands and their categories.
- `categories.csv`: Dataset containing product categories.

## Walkthrough Video

- [Walkthrough Video](https://github.com/yourusername/yourrepository/Walkthrough.mp4](https://github.com/Tarun-MR/Fetch/blob/main/Walkthrough.mp4))

## Troubleshooting

If you encounter any issues while running the tool, consider the following troubleshooting steps:

1. **Server Configuration:**
   - If you face difficulties, the likely culprit may be hosting on the same server. To address this, contemplate adjusting the server configuration in the final section of the code.
   - Specifically, alter the port value in the "port" variable to any number within the range of 8000 to 12000. Here's the relevant code snippet:

     ```python
     port = 9030  # Adjust the port if connection issues arise
     print(f" * Running on http://{host}:{port}/ (Press CTRL+C to quit)")
     app.run_server(debug=True, port=9030)
     ```
    make sure to update the port in app.run_server as well
2. **File Paths:**
   - Ensure that the required CSV files (`offer_retailer.csv`, `brand_category.csv`, `categories.csv`) are present in the specified file paths.
   - Check and provide the correct file paths if there are "file not found" errors.

3. **Error Handling:**
   - If you encounter any issues, refer to this troubleshooting section and ensure dependencies are correctly installed.

## Deployment Options

Consider different deployment options, such as deploying on a cloud platform or integrating with an existing system.

## Error Handling

If you encounter issues, refer to the troubleshooting section and ensure dependencies are correctly installed.

## User Guide

1. Enter a search query in the provided input field.
2. Click the "Submit" button to trigger the recommendation generation.
3. View the recommended offers and their similarity scores in the output section.

Feel free to explore and adapt the tool based on your evolving requirements. Collaboration is welcomed to refine and expand the tool further.
