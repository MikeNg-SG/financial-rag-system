from sentence_transformers import SentenceTransformer

def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer('all-MiniLM-L6-v2')

# Testing embedding model
if __name__ == "__main__":
    model = get_embedding_model()

    sentence_1 = "Total net sales increased 6% due primarily to growth in Services revenue."
    sentence_2 = "Revenue grew because more people bought subscriptions and apps."
    sentence_3 = "The weather in Cupertino was sunny yesterday."

    vec1 = model.encode(sentence_1)
    vec2 = model.encode(sentence_2)
    vec3 = model.encode(sentence_3)

    from sentence_transformers.util import cos_sim

    print("Sentence 1 vs 2 (both about revenue growth):", cos_sim(vec1, vec2).item())
    print("Sentence 1 vs 3 (revenue vs. unrelated weather):", cos_sim(vec1, vec3).item())