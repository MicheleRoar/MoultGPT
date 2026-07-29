def generate_trait_question_templates(trait):
    return [
        f"What does the article say about {trait}?",
        f"Is there any information regarding {trait}?",
        f"What findings are presented about {trait}?",
        f"Summarize the content related to {trait}.",
        f"How is {trait} discussed in the study?",
        f"What conclusions are drawn concerning {trait}?",
        f"What evidence supports the discussion of {trait}?"
    ]

def generate_combined_templates(traits):
    joined = ", ".join(traits)
    return [
        f"What does the article report about {joined}?",
        f"Can you summarize the findings on {joined}?",
        f"What is known about {joined} according to this study?",
        f"Describe any traits related to {joined} discussed in the paper.",
        f"Summarize the traits involving {joined}.",
        f"Does the article contain evidence on {joined}?",
        f"What moulting-related traits are associated with {joined}?"
    ]

def generate_negative_templates(trait):
    return [
        f"Is there any mention of {trait} in the article?",
        f"Does the paper provide information on {trait}?",
        f"Is {trait} covered in this study?",
        f"Can anything be inferred about {trait} from the article?",
        f"What does the article indicate about {trait}, if anything?"
    ]

positive_templates = [
    "Extract all moulting-related traits from the following article.",
    "What moulting-related traits are discussed in the article?",
    "Summarize all moulting phases, behaviours, and anatomical features.",
    "What do we learn about the moulting behaviour of this species?",
    "Which moulting-related transitions are observed?",
    "Describe all traits associated with moulting covered by the paper.",
    "What traits relevant to ecdysis, cuticle, or growth are reported?",
    "List and explain all moulting-related biological traits in the text.",
    "Provide a detailed account of all moulting-related information.",
    "Identify the full set of moulting traits mentioned in the article."
]

negative_outputs = [
    "The article does not mention this trait.",
    "No information is available on this trait in the text.",
    "This trait is not covered in the study.",
    "There is no discussion of this trait in the article.",
    "This topic is not addressed in the paper.",
    "No relevant content is found about this trait."
]