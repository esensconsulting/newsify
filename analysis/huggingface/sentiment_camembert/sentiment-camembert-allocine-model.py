from transformers import pipeline

def main():
    nlp = pipeline(task="sentiment-analysis", model="tblard/tf-allocine")

    examples = [
            "The movie was great!",
            "The movie was okay.",
            "The movie was terrible...",
            "The movie was terrific!",
            "Le film était pas terrible",
            "Le film était terrible !",
            "Le film était terriblement bien"
        ]

    for example in examples:
        print(example)
        print(nlp(example))

main()